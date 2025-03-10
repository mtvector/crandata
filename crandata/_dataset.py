"""_dataset.py – Dataset class for combining genome files and CrAnData objects
using a unified data_sources interface.
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from .crandata import CrAnData
from loguru import logger
from scipy.sparse import spmatrix
from tqdm import tqdm
from .crandata import LazyData
from ._genome import Genome
from .utils import one_hot_encode_sequence


def _flip_region_strand(region: str) -> str:
    strand_reverser = {"+": "-", "-": "+"}
    return region[:-1] + strand_reverser[region[-1]]

def _check_strandedness(region: str) -> bool:
    if re.fullmatch(r".+:\d+-\d+:[-+]", region):
        return True
    elif re.fullmatch(r".+:\d+-\d+", region):
        return False
    else:
        raise ValueError(
            f"Region {region} was not recognised as a valid coordinate set (chr:start-end or chr:start-end:strand)."
        )

def _deterministic_shift_region(region: str, stride: int = 50, n_shifts: int = 2) -> list[str]:
    new_regions = []
    chrom, start_end, strand = region.split(":")
    start, end = map(int, start_end.split("-"))
    for i in range(-n_shifts, n_shifts + 1):
        new_start = start + i * stride
        new_end = end + i * stride
        new_regions.append(f"{chrom}:{new_start}-{new_end}:{strand}")
    return new_regions

class SequenceLoader:
    def __init__(
        self,
        genome: Genome,
        in_memory: bool = False,
        always_reverse_complement: bool = False,
        deterministic_shift: bool = False,
        max_stochastic_shift: int = 0,
        regions: list[str] | None = None,
    ):
        self.genome = genome.fasta
        self.chromsizes = genome.chrom_sizes
        self.in_memory = in_memory
        self.always_reverse_complement = always_reverse_complement
        self.deterministic_shift = deterministic_shift
        self.max_stochastic_shift = max_stochastic_shift
        self.sequences = {}
        self.complement = str.maketrans("ACGT", "TGCA")
        self.regions = regions
        if self.in_memory:
            self._load_sequences_into_memory(self.regions)

    def _load_sequences_into_memory(self, regions: list[str]):
        logger.info("Loading sequences into memory...")
        stranded = _check_strandedness(regions[0])
        for region in tqdm(regions):
            if not stranded:
                region = f"{region}:+"
            regions_to_use = _deterministic_shift_region(region) if self.deterministic_shift else [region]
            for reg in regions_to_use:
                chrom, start_end, strand = reg.split(":")
                start, end = map(int, start_end.split("-"))
                extended_sequence = self._get_extended_sequence(chrom, start, end, strand)
                self.sequences[reg] = extended_sequence
                if self.always_reverse_complement:
                    self.sequences[_flip_region_strand(reg)] = self._reverse_complement(extended_sequence)

    def _get_extended_sequence(self, chrom: str, start: int, end: int, strand: str) -> str:
        extended_start = max(0, start - self.max_stochastic_shift)
        extended_end = extended_start + (end - start) + (self.max_stochastic_shift * 2)
        if self.chromsizes and chrom in self.chromsizes:
            chrom_size = self.chromsizes[chrom]
            if extended_end > chrom_size:
                extended_start = chrom_size - (end - start + self.max_stochastic_shift * 2)
                extended_end = chrom_size
        seq = self.genome.fetch(chrom, extended_start, extended_end).upper()
        if strand == "-":
            seq = self._reverse_complement(seq)
        return seq

    def _reverse_complement(self, sequence: str) -> str:
        return sequence.translate(self.complement)[::-1]

    def get_sequence(self, region: str, stranded: bool | None = None, shift: int = 0) -> str:
        if stranded is None:
            stranded = _check_strandedness(region)
        if not stranded:
            region = f"{region}:+"
        chrom, start_end, strand = region.split(":")
        start, end = map(int, start_end.split("-"))
        sequence = self.sequences[region] if self.in_memory else self._get_extended_sequence(chrom, start, end, strand)
        start_idx = self.max_stochastic_shift + shift
        end_idx = start_idx + (end - start)
        sub_sequence = sequence[start_idx:end_idx]
        if len(sub_sequence) < (end - start):
            sub_sequence = sub_sequence.ljust(end - start, "N") if strand == "+" else sub_sequence.rjust(end - start, "N")
        return sub_sequence

class IndexManager:
    def __init__(self, indices: list[str], always_reverse_complement: bool, deterministic_shift: bool = False):
        self.indices = indices
        self.always_reverse_complement = always_reverse_complement
        self.deterministic_shift = deterministic_shift
        self.augmented_indices, self.augmented_indices_map = self._augment_indices(indices)

    def shuffle_indices(self):
        np.random.shuffle(self.augmented_indices)

    def _augment_indices(self, indices: list[str]) -> tuple[list[str], dict[str, str]]:
        augmented_indices = []
        augmented_indices_map = {}
        for region in indices:
            stranded_region = region if _check_strandedness(region) else f"{region}:+"
            if self.deterministic_shift:
                shifted_regions = _deterministic_shift_region(stranded_region)
                for reg in shifted_regions:
                    augmented_indices.append(reg)
                    augmented_indices_map[reg] = region
                    if self.always_reverse_complement:
                        rc = _flip_region_strand(reg)
                        augmented_indices.append(rc)
                        augmented_indices_map[rc] = region
            else:
                augmented_indices.append(stranded_region)
                augmented_indices_map[stranded_region] = region
                if self.always_reverse_complement:
                    rc = _flip_region_strand(stranded_region)
                    augmented_indices.append(rc)
                    augmented_indices_map[rc] = region
        return augmented_indices, augmented_indices_map

if os.environ.get("KERAS_BACKEND") == "pytorch":
    import torch
    BaseClass = torch.utils.data.Dataset
else:
    BaseClass = object

class AnnDataset(BaseClass):
    """
    Dataset class for combining genome files and CrAnData objects using a unified data_sources interface.
    
    Parameters
    ----------
    adata : CrAnData
        CrAnData object containing the data.
    genome : Genome
        Genome instance.
    split : str, optional
        Split indicator (e.g. 'train', 'val', 'test').
    in_memory : bool
        Whether sequences are pre-loaded into memory.
    random_reverse_complement : bool
        Whether to randomly reverse complement sequences during training.
    always_reverse_complement : bool
        Whether to always augment sequences with their reverse complement.
    max_stochastic_shift : int
        Maximum random shift (in base pairs) applied during training.
    deterministic_shift : bool
        If True, use legacy fixed-stride shifting.
    data_sources : dict[str, str]
        Mapping of keys to data sources. Supported prefixes:
            "X", "layers/<key>", "varp/<key>", "obs/<col>", "obsm/<key>"
    """
    def __init__(
        self,
        adata: CrAnData,
        genome: Genome,
        split: str = None,
        in_memory: bool = True,
        random_reverse_complement: bool = False,
        always_reverse_complement: bool = False,
        max_stochastic_shift: int = 0,
        deterministic_shift: bool = False,
        data_sources: dict[str, str] = {'y': 'X'},
    ):
        try:
            super().__init__()
        except Exception:
            pass
        self.adata = adata
        # Convert X to xarray if needed:
        if not isinstance(self.adata.X, xr.DataArray):
            arr = np.asarray(self.adata.X)
            extra = arr.ndim - 2
            dims = ["obs", "var"] + [f"dim_{i}" for i in range(extra)]
            self.adata.X = xr.DataArray(arr, dims=dims)
        n_obs, n_var = self.adata.X.sizes["obs"], self.adata.X.sizes["var"]

        # Validate obs and var consistency
        if self.adata.obs is None:
            self.adata.obs = pd.DataFrame(index=[str(i) for i in range(n_obs)])
        if self.adata.var is None:
            self.adata.var = pd.DataFrame(index=[str(i) for i in range(n_var)])

        self.adata.X = self.adata.X.assign_coords(obs=("obs", np.array(self.adata.obs.index)),
                                                   var=("var", np.array(self.adata.var.index)))
        self.uns = self.adata.uns if self.adata.uns is not None else {}
        self.layers = self.adata.layers if self.adata.layers is not None else {}
        self.data_sources = data_sources

        self.compressed = isinstance(self.adata.X, spmatrix)
        self.indices = list(self.adata.var_names)
        self.index_map = {index: i for i, index in enumerate(self.indices)}
        self.num_outputs = self.adata.X.shape[0]
        self.random_reverse_complement = random_reverse_complement
        self.max_stochastic_shift = max_stochastic_shift
        self.meta_obs_names = np.array(self.adata.obs_names)
        self.shuffle = False

        self.sequence_loader = SequenceLoader(
            genome,
            in_memory=in_memory,
            always_reverse_complement=always_reverse_complement,
            deterministic_shift=deterministic_shift,
            max_stochastic_shift=max_stochastic_shift,
            regions=self.indices,
        )
        self.index_manager = IndexManager(
            self.indices,
            always_reverse_complement=always_reverse_complement,
            deterministic_shift=deterministic_shift,
        )
        self.region_width = (
            self.adata.uns.get('params', {}).get('target_region_width',
                int(np.round(np.mean(self.adata.var['end'] - self.adata.var['start']))) - (2 * self.max_stochastic_shift)
            )
        )

        # Set augmented probabilities based on split.
        if split in ['train', 'val', 'test', 'predict']:
            probs = self.adata.var[f"{split}_probs"].values.astype(float)
        else:
            probs = np.ones(self.adata.shape[1])
        probs = np.clip(probs, 0, None)
        n_aug = len(self.index_manager.augmented_indices)
        self.augmented_probs = np.ones(n_aug, dtype=float)
        self.augmented_probs /= self.augmented_probs.sum()
        for i, aug_region in enumerate(self.index_manager.augmented_indices):
            original_region = self.index_manager.augmented_indices_map[aug_region]
            var_idx = self.index_map[original_region]
            self.augmented_probs[i] = probs[var_idx]

    def __getitem__(self, idx: int) -> dict:
        if not isinstance(idx, int):
            raise IndexError("Index must be an integer.")
        augmented_index = self.index_manager.augmented_indices[idx]
        original_index = self.index_manager.augmented_indices_map[augmented_index]
        shift = (np.random.randint(-self.max_stochastic_shift, self.max_stochastic_shift + 1)
                 if self.max_stochastic_shift > 0 else 0)
        x_seq = self.sequence_loader.get_sequence(augmented_index, stranded=True, shift=shift)
        if self.random_reverse_complement and np.random.rand() < 0.5:
            x_seq = self.sequence_loader._reverse_complement(x_seq)
        seq_onehot = one_hot_encode_sequence(x_seq, expand_dim=False)
        item = {"sequence": seq_onehot}
        for key, source_str in self.data_sources.items():
            if key == "sequence":
                continue
            if source_str in ["X"] or source_str.startswith("layers/") or source_str.startswith("varp/"):
                # Return a LazyData proxy
                item[key] = self._get_data_array(source_str, original_index, shift=shift)
            elif source_str.startswith("obs/"):
                col = source_str.split("/", 1)[1]
                item[key] = self.adata.obs[col].values
            elif source_str.startswith("obsm/"):
                key_name = source_str.split("/", 1)[1]
                item[key] = self.adata.obsm[key_name].values
            else:
                raise ValueError(f"Unknown data source prefix in '{source_str}'")
        return item

    def _get_data_array(self, source_str: str, varname: str, shift: int = 0):
        var_idx = self.index_map[varname]
        if source_str == "X":
            lazy_obj = self.adata.X.data  # Might be LazyH5Array OR already an array.
            # Only wrap in LazyData if the underlying object is truly lazy.
            if hasattr(lazy_obj, "filename"):
                key = (slice(None), var_idx)
                return LazyData(lazy_obj, key,
                                local_obs=np.array(self.adata.obs.index),
                                global_obs=np.array(self.adata.meta_obs_names))
            else:
                # If it's already an array, return the appropriate slice.
                return self.adata.X[:, var_idx]
        elif source_str.startswith("layers/"):
            key_name = source_str.split("/", 1)[1]
            start_idx = self.max_stochastic_shift + shift
            end_idx = start_idx + self.region_width
            if hasattr(self.adata.layers[key_name], "data"):
                lazy_obj = self.adata.layers[key_name].data
                if hasattr(lazy_obj, "filename"):
                    key_tuple = (self.meta_obs_names, var_idx, slice(start_idx, end_idx))
                    return LazyData(lazy_obj, key_tuple,
                                    local_obs=np.array(self.adata.obs.index),
                                    global_obs=np.array(self.adata.meta_obs_names))
            # Otherwise, fall back to loading immediately.
            return self.adata.layers[key_name][self.meta_obs_names, var_idx][..., start_idx:end_idx]
        elif source_str.startswith("varp/"):
            key_name = source_str.split("/", 1)[1]
            if hasattr(self.adata.varp[key_name], "data"):
                lazy_obj = self.adata.varp[key_name].data
                if hasattr(lazy_obj, "filename"):
                    key_tuple = (var_idx,)
                    return LazyData(lazy_obj, key_tuple)
            return self.adata.varp[key_name][var_idx]
        else:
            raise ValueError(f"Data source '{source_str}' is not indexable by variable.")

    def __call__(self):
        for i in range(len(self)):
            if i == 0 and self.shuffle:
                self.index_manager.shuffle_indices()
            yield self.__getitem__(i)

    def __len__(self) -> int:
        return len(self.index_manager.augmented_indices)

    def __repr__(self) -> str:
        return (f"AnnDataset(anndata_shape={self.adata.shape}, n_samples={len(self)}, "
                f"num_outputs={self.num_outputs}, split={self.adata.var.get('split', 'None')}, "
                f"in_memory={self.adata.X is not None})")

class MetaAnnDataset:
    def __init__(self, datasets: list[AnnDataset]):
        if not datasets:
            raise ValueError("No AnnDataset provided.")
        self.datasets = datasets
        self.global_indices = []
        self.global_probs = []
        for ds_idx, ds in enumerate(datasets):
            ds_len = len(ds.index_manager.augmented_indices)
            if ds_len == 0:
                continue
            if ds.augmented_probs is not None:
                for local_i in range(ds_len):
                    self.global_indices.append((ds_idx, local_i))
                    self.global_probs.append(ds.augmented_probs[local_i])
            else:
                for local_i in range(ds_len):
                    self.global_indices.append((ds_idx, local_i))
                    self.global_probs.append(1.0)
        self.global_probs = np.array(self.global_probs, dtype=float)
        total = self.global_probs.sum()
        if total > 0:
            self.global_probs /= total
        else:
            self.global_probs.fill(1.0 / len(self.global_probs))

    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, global_idx):
        # If global_idx is a tuple, assume it's already (ds_idx, local_i)
        if isinstance(global_idx, tuple):
            ds_idx, local_i = global_idx
        else:
            ds_idx, local_i = self.global_indices[global_idx]
        return self.datasets[int(ds_idx)][int(local_i)]

    def __repr__(self):
        return (f"MetaAnnDataset(num_datasets={len(self.datasets)}, total_augmented_indices={len(self.global_indices)})")
