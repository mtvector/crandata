"""chrom_io.py – Creating AnnDataModule from bigwigs."""

from __future__ import annotations
import os
import re
import tempfile
from pathlib import Path
import numpy as np
import pybigtools
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
import xarray as xr
import sparse
import h5py  # for HDF5 backing
from . import _conf
from .crandata import CrAnData

# -----------------------
# Utility functions
# -----------------------

def _sort_files(filename: Path):
    filename = Path(filename)
    parts = filename.stem.split("_")
    if len(parts) > 1:
        try:
            return (False, int(parts[1]))
        except ValueError:
            return (True, filename.stem)
    return (True, filename.stem)

def _custom_region_sort(region: str) -> tuple[int, int, int]:
    chrom, pos = region.split(":")
    start, _ = map(int, pos.split("-"))
    numeric_match = re.match(r"chr(\d+)|chrom(\d+)", chrom, re.IGNORECASE)
    if numeric_match:
        chrom_num = int(numeric_match.group(1) or numeric_match.group(2))
        return (0, chrom_num, start)
    else:
        return (1, chrom, start)

def _read_chromsizes(chromsizes_file: Path) -> dict[str, int]:
    # We still use pandas here for convenience, but its output is only used transiently.
    import pandas as pd
    chromsizes = pd.read_csv(chromsizes_file, sep="\t", header=None, names=["chrom", "size"])
    return chromsizes.set_index("chrom")["size"].to_dict()

def _extract_values_from_bigwig(bw_file: Path, bed_file: Path, target: str, n_bins: int = None) -> np.ndarray:
    bw_file = str(bw_file)
    bed_file = str(bed_file)
    with pybigtools.open(bw_file, "r") as bw:
        chromosomes_in_bigwig = set(bw.chroms())
    temp_bed_file = tempfile.NamedTemporaryFile(delete=False)
    bed_entries_to_keep_idx = []
    with open(bed_file) as fh:
        for idx, line in enumerate(fh):
            chrom = line.split("\t", 1)[0]
            if chrom in chromosomes_in_bigwig:
                temp_bed_file.write(line.encode("utf-8"))
                bed_entries_to_keep_idx.append(idx)
    temp_bed_file.close()
    total_bed_entries = idx + 1
    bed_entries_to_keep_idx = np.array(bed_entries_to_keep_idx, np.intp)
    
    if target == "mean":
        with pybigtools.open(bw_file, "r") as bw:
            values = np.fromiter(
                bw.average_over_bed(bed=temp_bed_file.name, names=None, stats="mean0"),
                dtype=np.float32,
            )
    elif target == "max":
        with pybigtools.open(bw_file, "r") as bw:
            values = np.fromiter(
                bw.average_over_bed(bed=temp_bed_file.name, names=None, stats="max"),
                dtype=np.float32,
            )
    elif target == "count":
        with pybigtools.open(bw_file, "r") as bw:
            values = np.fromiter(
                bw.average_over_bed(bed=temp_bed_file.name, names=None, stats="sum"),
                dtype=np.float32,
            )
    elif target == "logcount":
        with pybigtools.open(bw_file, "r") as bw:
            values = np.log1p(
                np.fromiter(
                    bw.average_over_bed(bed=temp_bed_file.name, names=None, stats="sum"),
                    dtype=np.float32,
                )
            )
    elif target == "raw":
        with pybigtools.open(bw_file, "r") as bw:
            lines = open(temp_bed_file.name).readlines()
            values_list = [
                np.array(
                    bw.values(chrom, int(start), int(end), missing=0., exact=False, bins=n_bins, summary='mean')
                )
                for chrom, start, end in [line.split("\t")[:3] for line in lines]
            ]
            values = np.vstack(values_list)
    else:
        raise ValueError(f"Unsupported target '{target}'")
    os.remove(temp_bed_file.name)
    if target == "raw":
        all_data = np.full((total_bed_entries, values.shape[1]), np.nan, dtype=np.float32)
        all_data[bed_entries_to_keep_idx, :] = values
        return all_data
    else:
        if values.shape[0] != total_bed_entries:
            all_values = np.full(total_bed_entries, np.nan, dtype=np.float32)
            all_values[bed_entries_to_keep_idx] = values
            return all_values
        else:
            return values

def _read_consensus_regions(regions_file: Path, chromsizes_dict: dict | None = None):
    # Use pandas temporarily to read the BED file, then convert to numpy arrays.
    import pandas as pd
    if chromsizes_dict is None and not _conf.genome:
        logger.warning("Chromsizes file not provided. Will not check if regions are within chromosomes", stacklevel=1)
    consensus_peaks = pd.read_csv(
        regions_file,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        dtype={0: str, 1: "Int32", 2: "Int32"},
    )
    consensus_peaks.columns = ["chrom", "start", "end"]
    consensus_peaks["region"] = consensus_peaks["chrom"].astype(str) + ":" + consensus_peaks["start"].astype(str) + "-" + consensus_peaks["end"].astype(str)
    if chromsizes_dict:
        pass
    elif _conf.genome:
        chromsizes_dict = _conf.genome.chrom_sizes
    else:
        return consensus_peaks
    valid_mask = consensus_peaks.apply(
        lambda row: row["chrom"] in chromsizes_dict and row["start"] >= 0 and row["end"] <= chromsizes_dict[row["chrom"]],
        axis=1,
    )
    consensus_peaks_filtered = consensus_peaks[valid_mask]
    if len(consensus_peaks) != len(consensus_peaks_filtered):
        logger.warning(f"Filtered {len(consensus_peaks) - len(consensus_peaks_filtered)} consensus regions (not within chromosomes)")
    return consensus_peaks_filtered

def _create_temp_bed_file(consensus_peaks, target_region_width: int, adjust=True) -> str:
    # Using the pandas DataFrame from _read_consensus_regions here.
    adjusted_peaks = consensus_peaks.copy()
    if adjust:
        adjusted_peaks[1] = adjusted_peaks.apply(
            lambda row: max(0, row.iloc[1] - (target_region_width - (row.iloc[2] - row.iloc[1])) // 2),
            axis=1,
        )
        adjusted_peaks[2] = adjusted_peaks[1] + target_region_width
        adjusted_peaks[1] = adjusted_peaks[1].astype(int)
        adjusted_peaks[2] = adjusted_peaks[2].astype(int)
    temp_bed_file = "temp_adjusted_regions.bed"
    adjusted_peaks.to_csv(temp_bed_file, sep="\t", header=False, index=False)
    return temp_bed_file

def _check_bed_file_format(bed_file: Path) -> None:
    with open(bed_file) as f:
        first_line = f.readline().strip()
    if len(first_line.split("\t")) < 3:
        raise ValueError(f"BED file '{bed_file}' is not in the correct format. Expected at least three tab-separated columns.")
    pattern = r".*\t\d+\t\d+.*"
    if not re.match(pattern, first_line):
        raise ValueError(f"BED file '{bed_file}' is not in the correct format. Expected columns 2 and 3 to contain integers.")

def _filter_and_adjust_chromosome_data(peaks, chrom_sizes: dict,
                                      max_shift: int = 0, chrom_col: str = "chrom",
                                      start_col: str = "start", end_col: str = "end",
                                      MIN_POS: int = 0):
    peaks["_chr_size"] = peaks[chrom_col].map(chrom_sizes)
    peaks = peaks.dropna(subset=["_chr_size"]).copy()
    peaks["_chr_size"] = peaks["_chr_size"].astype(int)
    starts = peaks[start_col].to_numpy(dtype=int)
    ends = peaks[end_col].to_numpy(dtype=int)
    chr_sizes_arr = peaks["_chr_size"].to_numpy(dtype=int)
    orig_length = ends - starts
    desired_length = orig_length + 2 * max_shift
    new_starts = starts - max_shift
    new_ends = new_starts + desired_length
    cond_left_edge = new_starts < MIN_POS
    shift_needed = MIN_POS - new_starts[cond_left_edge]
    new_starts[cond_left_edge] = MIN_POS
    new_ends[cond_left_edge] += shift_needed
    cond_right_edge = new_ends > chr_sizes_arr
    shift_needed = new_ends[cond_right_edge] - chr_sizes_arr[cond_right_edge]
    new_ends[cond_right_edge] = chr_sizes_arr[cond_right_edge]
    new_starts[cond_right_edge] -= shift_needed
    cond_left_clamp = new_starts < MIN_POS
    new_starts[cond_left_clamp] = MIN_POS
    peaks[start_col] = new_starts
    peaks[end_col] = new_ends
    peaks.drop(columns=["_chr_size"], inplace=True)
    return peaks

# -----------------------
# X array writing
# -----------------------

def _load_x_to_memory(bw_files, consensus_peaks, target, target_region_width,
                      out_path, obs_index, var_index, chunk_size=1024, n_bins=None) -> xr.DataArray:
    """
    Write training data (extracted from bigWig files) to an HDF5 file.
    The final shape is (n_obs, n_var, seq_len).
    """
    n_obs = len(bw_files)
    n_var = consensus_peaks.shape[0]
    
    # Determine sequence length using the first file.
    temp_bed = _create_temp_bed_file(consensus_peaks, target_region_width)
    sample = _extract_values_from_bigwig(bw_files[0], temp_bed, target=target, n_bins=n_bins)
    os.remove(temp_bed)
    if sample.ndim == 1:
        sample = sample.reshape(n_var, 1)
    seq_len = sample.shape[1]
    chunk_size = min(chunk_size, n_var)
    
    with h5py.File(out_path, "w") as f:
        dset = f.create_dataset("X", shape=(n_obs, n_var, seq_len),
                                chunks=(n_obs, chunk_size, seq_len),
                                dtype="float32",
                                fillvalue=np.nan)
        for i, bw_file in tqdm(enumerate(bw_files)):
            temp_bed = _create_temp_bed_file(consensus_peaks, target_region_width)
            result = _extract_values_from_bigwig(bw_file, temp_bed, target=target, n_bins=n_bins)
            os.remove(temp_bed)
            if result.ndim == 1:
                result = result.reshape(n_var, 1)
            dset[i, :, :] = result
    
    with h5py.File(out_path, "r") as f:
        X_array = f["X"][:]
    X = xr.DataArray(X_array, dims=["obs", "var", "seq_len"],
                     coords={"obs": np.array(obs_index),
                             "var": np.array(var_index),
                             "seq_len": np.arange(seq_len)})
    return X

# -----------------------
# Main import function
# -----------------------

def import_bigwigs(bigwigs_folder: Path, regions_file: Path,
                   backed_path: Path, target_region_width: int | None,
                   target: str = 'raw',  # e.g. "raw", "mean", etc.
                   chromsizes_file: Path | None = None, genome: any = None,
                   max_stochastic_shift: int = 0, chunk_size: int = 512, n_bins: int = None) -> CrAnData:
    """
    Import bigWig files and consensus regions to create a backed CrAnData object.
    """
    bigwigs_folder = Path(bigwigs_folder)
    regions_file = Path(regions_file)
    if not bigwigs_folder.is_dir():
        raise FileNotFoundError(f"Directory '{bigwigs_folder}' not found")
    if not regions_file.is_file():
        raise FileNotFoundError(f"File '{regions_file}' not found")
    
    if chromsizes_file is not None:
        chromsizes_dict = _read_chromsizes(chromsizes_file)
    if genome is not None:
        chromsizes_dict = genome.chrom_sizes
    
    _check_bed_file_format(regions_file)
    consensus_peaks = _read_consensus_regions(regions_file, chromsizes_dict)
    region_width = int(np.round(np.mean(consensus_peaks['end'] - consensus_peaks['start'])))
    consensus_peaks = consensus_peaks.loc[(consensus_peaks['end'] - consensus_peaks['start']) == region_width, :]
    consensus_peaks = _filter_and_adjust_chromosome_data(consensus_peaks, chromsizes_dict, max_shift=max_stochastic_shift)
    shifted_width = target_region_width + 2 * max_stochastic_shift
    consensus_peaks = consensus_peaks.loc[(consensus_peaks['end'] - consensus_peaks['start']) == shifted_width, :]
    
    bw_files = []
    chrom_set = set()
    for file in tqdm(os.listdir(bigwigs_folder)):
        file_path = os.path.join(bigwigs_folder, file)
        try:
            bw = pybigtools.open(file_path, "r")
            chrom_set |= set(bw.chroms().keys())
            bw_files.append(file_path)
            bw.close()
        except (ValueError, pybigtools.BBIReadError):
            pass
    consensus_peaks = consensus_peaks.loc[consensus_peaks["chrom"].isin(chrom_set), :]
    bw_files = sorted([os.path.join(bigwigs_folder, f) for f in os.listdir(bigwigs_folder)])
    if not bw_files:
        raise FileNotFoundError(f"No valid bigWig files found in '{bigwigs_folder}'")
    
    logger.info(f"Extracting values from {len(bw_files)} bigWig files...")
    # Instead of creating pandas DataFrames, we create numpy arrays and wrap them in xarray DataArrays.
    obs_index = np.array([os.path.basename(f).rpartition(".")[0].replace(".", "_") for f in bw_files])
    obs_array = xr.DataArray(obs_index, dims=["obs"])
    obs_metadata = {"file_path": np.array(bw_files), "obs": obs_array}
    
    var_index = consensus_peaks["region"].to_numpy()
    chunk_index = np.arange(consensus_peaks.shape[0]) // chunk_size
    var_metadata = {
        "chr": consensus_peaks["chrom"].to_numpy(),
        "start": consensus_peaks["start"].to_numpy(),
        "end": consensus_peaks["end"].to_numpy(),
        "region": var_index,
        "chunk_index": chunk_index
    }
    
    X = _load_x_to_memory(
            bw_files, consensus_peaks, target, target_region_width,
            out_path=str(backed_path), obs_index=obs_index, var_index=var_index,
            chunk_size=chunk_size, n_bins=n_bins)
    
    # Create the CrAnData object without any pandas DataFrames.
    adata = CrAnData(X=X, obs=obs_metadata, var=var_metadata, uns=None, obsm=None, varm=None, layers=None,
                     axis_indices={"obs": obs_index, "var": var_index},
                     global_axis_order=["var", "obs"])
    adata.uns = {'params': {
        'target_region_width': target_region_width,
        'shifted_region_width': target_region_width + 2 * max_stochastic_shift,
        'max_stochastic_shift': max_stochastic_shift,
        'chunk_size': chunk_size
    }}
    adata.to_h5(str(backed_path))
    adata = CrAnData.from_h5(backed_path, backed=["X"])
    return adata

# -----------------------
# Additional utility functions
# -----------------------

def prepare_intervals(adata):
    """
    Prepare sorted interval data structures from adata.var.
    Assumes adata.var is a dictionary of arrays.
    """
    # Here we assume that adata.var["chr"], ["start"], and ["end"] are available as numpy arrays.
    chr_arr = np.array(adata.var["chr"])
    start_arr = np.array(adata.var["start"])
    end_arr = np.array(adata.var["end"])
    # Create a list of region names from the "region" array.
    var_names = np.array(adata.var["region"])
    
    intervals = defaultdict(list)
    # Sort by chromosome and start.
    order = np.lexsort((start_arr, chr_arr))
    for idx in order:
        intervals[chr_arr[idx]].append((start_arr[idx], end_arr[idx], var_names[idx]))
    return dict(intervals)

def _find_overlaps_in_sorted_bed(bed_df, chrom_intervals):
    # (Unchanged: assumes bed_df is created from external files)
    row_to_overlaps = defaultdict(list)
    chrom_positions = defaultdict(int)
    for _, row in bed_df.iterrows():
        chrom = row["chr"]
        start_q = row["start"]
        end_q = row["end"]
        row_idx = row["row_idx"]
        intervals = chrom_intervals.get(chrom, [])
        pos = chrom_positions.get(chrom, 0)
        while pos < len(intervals) and intervals[pos][1] < start_q:
            pos += 1
        scan_pos = pos
        while scan_pos < len(intervals) and intervals[scan_pos][0] < end_q:
            (_, _, var_name) = intervals[scan_pos]
            row_to_overlaps[row_idx].append(var_name)
            scan_pos += 1
        chrom_positions[chrom] = scan_pos
    return row_to_overlaps

def _find_overlaps_for_bedp(bedp_df, chrom_intervals, coord_col_prefix):
    df = bedp_df[[f"{coord_col_prefix}", f"start{coord_col_prefix[-1]}", f"end{coord_col_prefix[-1]}", "row_idx"]].copy()
    df = df.rename(columns={
        f"{coord_col_prefix}": "chr",
        f"start{coord_col_prefix[-1]}": "start",
        f"end{coord_col_prefix[-1]}": "end"
    })
    df = df.sort_values(["chr", "start"]).reset_index(drop=True)
    return _find_overlaps_in_sorted_bed(df, chrom_intervals)

def add_contact_strengths_to_varp(adata, bedp_files, key="hic_contacts"):
    """
    Read Hi-C BEDP files and add a Hi-C contact data array into adata.varp[key].
    """
    # Here we assume that adata.var is a dictionary of numpy arrays.
    if "chr" not in adata.var:
        # Fallback conversion if needed.
        adata.var["chr"] = np.array([x.split(":")[0] for x in adata.var["region"]])
        adata.var["start"] = np.array([int(x.split(":")[1].split("-")[0]) for x in adata.var["region"]])
        adata.var["end"] = np.array([int(x.split(":")[1].split("-")[1]) for x in adata.var["region"]])
    
    chrom_intervals = prepare_intervals(adata)
    num_bins = adata.var["region"].shape[0]
    num_files = len(bedp_files)
    var_names = adata.var["region"]
    var_name_to_i = {var: i for i, var in enumerate(var_names)}
    
    all_rows = []
    all_cols = []
    all_file_idx = []
    all_data = []
    
    import pandas as pd
    for fidx, bedp_file in enumerate(bedp_files):
        bedp_df = pd.read_csv(
            bedp_file, sep="\t", header=None,
            names=["chr1", "start1", "end1", "chr2", "start2", "end2", "score"]
        ).reset_index(drop=True)
        bedp_df["row_idx"] = bedp_df.index
        
        bedp_first = (
            bedp_df[["chr1", "start1", "end1", "row_idx"]]
            .rename(columns={"chr1": "chr", "start1": "start", "end1": "end"})
            .sort_values(["chr", "start"])
            .reset_index(drop=True)
        )
        bedp_second = (
            bedp_df[["chr2", "start2", "end2", "row_idx"]]
            .rename(columns={"chr2": "chr", "start2": "start", "end2": "end"})
            .sort_values(["chr", "start"])
            .reset_index(drop=True)
        )
        
        overlaps_first = _find_overlaps_in_sorted_bed(bedp_first, chrom_intervals)
        overlaps_second = _find_overlaps_in_sorted_bed(bedp_second, chrom_intervals)
        
        for row_idx, row in bedp_df.iterrows():
            ovs1 = overlaps_first.get(row_idx, [])
            ovs2 = overlaps_second.get(row_idx, [])
            if not ovs1 or not ovs2:
                continue
            ovs1_int = [var_name_to_i[v] for v in ovs1 if v in var_name_to_i]
            ovs2_int = [var_name_to_i[v] for v in ovs2 if v in var_name_to_i]
            for i_idx in ovs1_int:
                for j_idx in ovs2_int:
                    all_rows.append(i_idx)
                    all_cols.append(j_idx)
                    all_file_idx.append(fidx)
                    all_data.append(float(row["score"]))
    
    if all_rows:
        all_rows = np.array(all_rows, dtype=np.int64)
        all_cols = np.array(all_cols, dtype=np.int64)
        all_file_idx = np.array(all_file_idx, dtype=np.int64)
        all_data = np.array(all_data, dtype=np.float32)
    else:
        all_rows = np.array([0], dtype=np.int64)
        all_cols = np.array([0], dtype=np.int64)
        all_file_idx = np.array([0], dtype=np.int64)
        all_data = np.array([0], dtype=np.float32)
    
    size = (num_bins, num_bins, num_files)
    indices = np.stack([all_rows, all_cols, all_file_idx], axis=0)
    from sparse import COO
    contacts_tensor = COO(indices, all_data, shape=size)
    
    contacts_xr = xr.DataArray(
        contacts_tensor,
        dims=["var", "var_target", "obs"],
        coords={
            "var": var_names,
            "var_target": var_names,
            "obs": np.array([os.path.basename(x).replace('.bedp','') for x in bedp_files])
        }
    )
    
    if "varp" not in adata._data or adata._data["varp"] is None:
        adata._data["varp"] = {}
        adata._create_dynamic_property("varp", convert_to_dataframe=False)
    adata._data["varp"][key] = contacts_xr
    adata.add_property("varp", adata._data["varp"])
    
    return contacts_xr
