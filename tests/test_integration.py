import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pybigtools  # Using pybigtools instead of pyBigWig
import xarray as xr

from crandata.chrom_io import import_bigwigs, add_contact_strengths_to_varp
from crandata._genome import Genome
from crandata._anndatamodule import MetaAnnDataModule
from crandata.crandata import CrAnData

from crandata.chrom_io import import_bigwigs
from crandata._anndatamodule import MetaAnnDataModule
from crandata.utils import hot_encoding_to_sequence


# -----------------------------------------------------------------------------
# Fixtures for temporary test setup
# -----------------------------------------------------------------------------

@pytest.fixture
def temp_setup(tmp_path: Path):
    """
    Set up a temporary directory structure with necessary files:
      - A base directory with subdirectories for beds and bigwigs.
      - A chromsizes file.
      - A consensus BED file.
      - A simple BigWig file (created with pybigtools).
    """
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    beds_dir = base_dir / "beds"
    bigwigs_dir = base_dir / "bigwigs"
    beds_dir.mkdir()
    bigwigs_dir.mkdir()

    # Create a chromsizes file
    chromsizes_file = base_dir / "chrom.sizes"
    chromsizes_file.write_text("chr1\t1000\n")

    # Create two example BED files (for context)
    bed_data_A = pd.DataFrame({0: ["chr1", "chr1"],
                               1: [100, 300],
                               2: [200, 400]})
    bed_data_B = pd.DataFrame({0: ["chr1", "chr1"],
                               1: [150, 350],
                               2: [250, 450]})
    (beds_dir / "ClassA.bed").write_text(bed_data_A.to_csv(sep="\t", header=False, index=False))
    (beds_dir / "ClassB.bed").write_text(bed_data_B.to_csv(sep="\t", header=False, index=False))

    # Create a consensus BED file
    consensus = pd.DataFrame({0: ["chr1", "chr1", "chr1"],
                              1: [100, 300, 350],
                              2: [200, 400, 450]})
    consensus_file = base_dir / "consensus.bed"
    consensus_file.write_text(consensus.to_csv(sep="\t", header=False, index=False))

    # Create a simple BigWig file with a single chromosome region using pybigtools.
    bigwig_file = bigwigs_dir / "test.bw"
    bw = pybigtools.open(str(bigwig_file), mode="w")
    bw.write(chroms={"chr1": 1000}, vals=[("chr1", 0, 1000, 5.0)])
    bw.close()

    # Path for HDF5 backing file (will be created during import)
    backed_path = base_dir / "chrom_data.h5"

    return {
        "base_dir": base_dir,
        "beds_dir": beds_dir,
        "bigwigs_dir": bigwigs_dir,
        "chromsizes_file": chromsizes_file,
        "consensus_file": consensus_file,
        "bigwig_file": bigwig_file,
        "backed_path": backed_path,
    }

# -----------------------------------------------------------------------------
# Tests for basic functionality
# -----------------------------------------------------------------------------

def test_import_bigwigs(temp_setup):
    """
    Test that import_bigwigs creates a CrAnData object from the provided BigWig and consensus files.
    """
    target_region_width = 100
    adata = import_bigwigs(
        bigwigs_folder=temp_setup["bigwigs_dir"],
        regions_file=temp_setup["consensus_file"],
        backed_path=str(temp_setup["backed_path"]),
        target_region_width=target_region_width,
        chromsizes_file=str(temp_setup["chromsizes_file"]),
    )
    # Basic assertions
    assert isinstance(adata, CrAnData)
    assert adata.obs.shape[0] > 0
    assert adata.var.shape[0] > 0
    assert os.path.exists(str(temp_setup["backed_path"]))

def test_add_contact_strengths_to_varp(temp_setup):
    """
    Test that a synthetic Hi-C BEDP file is correctly processed and the contacts data
    are added to adata.varp.
    """
    target_region_width = 100
    adata = import_bigwigs(
        bigwigs_folder=temp_setup["bigwigs_dir"],
        regions_file=temp_setup["consensus_file"],
        backed_path=str(temp_setup["backed_path"]),
        target_region_width=target_region_width,
        chromsizes_file=str(temp_setup["chromsizes_file"]),
    )
    # Create synthetic BEDP data for Hi-C contacts
    synthetic_bedp = pd.DataFrame({
        0: ["chr1", "chr1"],
        1: [100, 300],
        2: [200, 400],
        3: ["chr1", "chr1"],
        4: [150, 350],
        5: [250, 450],
        6: [10, 20]
    })
    synthetic_bedp_file = temp_setup["base_dir"] / "synthetic.bedp"
    synthetic_bedp_file.write_text(synthetic_bedp.to_csv(sep="\t", header=False, index=False))
    
    contacts = add_contact_strengths_to_varp(adata, [str(synthetic_bedp_file)], key="hic_contacts")
    # Check that the contacts data are stored in varp and is an xarray DataArray
    assert "hic_contacts" in adata.varp
    assert isinstance(adata.varp["hic_contacts"], xr.DataArray)

def test_h5_save_load(temp_setup):
    """
    Test that a CrAnData object can be written to an HDF5 file and then re-loaded
    with all key components (obs, var, obsm) intact.
    """
    target_region_width = 100
    adata = import_bigwigs(
        bigwigs_folder=temp_setup["bigwigs_dir"],
        regions_file=temp_setup["consensus_file"],
        backed_path=str(temp_setup["backed_path"]),
        target_region_width=target_region_width,
        chromsizes_file=str(temp_setup["chromsizes_file"]),
    )
    # Add a random obsm entry
    adata.obsm['gex'] = xr.DataArray(np.random.randn(adata.obs.shape[0], 100),
                                     dims=['types', 'genes'])
    h5_path = temp_setup["base_dir"] / "adata.h5"
    adata.to_h5(str(h5_path))
    adata_loaded = CrAnData.from_h5(str(h5_path))
    # Verify that the loaded object has matching obs and var DataFrames
    pd.testing.assert_frame_equal(adata.obs, adata_loaded.obs)
    pd.testing.assert_frame_equal(adata.var, adata_loaded.var)
    assert 'gex' in adata_loaded.obsm

def test_meta_ann_data_module(temp_setup, tmp_path: Path):
    """
    Test the creation and basic operation of a MetaAnnDataModule:
      - Create two copies of a CrAnData object (adding a 'split' column).
      - Create a dummy FASTA file and Genome object.
      - Populate Hi-C contact data in adata.varp using synthetic BEDP data.
      - Build the MetaAnnDataModule and check that batches from the training dataloader
        have the expected keys and batch dimensions.
    """
    target_region_width = 100
    adata = import_bigwigs(
        bigwigs_folder=temp_setup["bigwigs_dir"],
        regions_file=temp_setup["consensus_file"],
        backed_path=str(temp_setup["backed_path"]),
        target_region_width=target_region_width,
        chromsizes_file=str(temp_setup["chromsizes_file"]),
    )
    # Add a random obsm entry and a 'split' column in var
    adata.obsm['gex'] = xr.DataArray(np.random.randn(adata.obs.shape[0], 100),
                                     dims=['types', 'genes'])
    adata.var["split"] = "train"
    
    # --- New: Create synthetic Hi-C BEDP data and add it to adata.varp ---
    synthetic_bedp = pd.DataFrame({
        0: ["chr1", "chr1"],
        1: [100, 300],
        2: [200, 400],
        3: ["chr1", "chr1"],
        4: [150, 350],
        5: [250, 450],
        6: [10, 20]
    })
    synthetic_bedp_file = temp_setup["base_dir"] / "synthetic.bedp"
    synthetic_bedp_file.write_text(synthetic_bedp.to_csv(sep="\t", header=False, index=False))
    from crandata.chrom_io import add_contact_strengths_to_varp
    add_contact_strengths_to_varp(adata, [str(synthetic_bedp_file)], key="hic_contacts")
    # ---------------------------------------------------------------------
    
    # Create a dummy FASTA file for the genome
    fasta_file = tmp_path / "chr1.fa"
    fasta_file.write_text(">chr1\n" + "A" * 1000 + "\n")
    dummy_genome = Genome(str(fasta_file), chrom_sizes=str(temp_setup["chromsizes_file"]))
    
    # Create two copies of the CrAnData object to simulate two datasets
    adata1 = adata.copy()
    adata2 = adata.copy()
    
    # Instantiate MetaAnnDataModule with both datasets and corresponding genomes.
    meta_module = MetaAnnDataModule(
        adatas=[adata1, adata2],
        genomes=[dummy_genome, dummy_genome],
        data_sources={'y': 'X', 'hic': 'varp/hic_contacts', 'gex': 'obsm/gex'},
        in_memory=True,
        random_reverse_complement=True,
        max_stochastic_shift=5,
        deterministic_shift=False,
        shuffle_obs=True,
        shuffle=True,
        batch_size=3,    # small batch size for testing
        epoch_size=10
    )
    meta_module.setup("fit")
    train_dl = meta_module.train_dataloader
    
    # Get one batch from the dataloader and check for expected keys and shapes
    batch = next(iter(train_dl.data))
    for key in ['sequence', 'y', 'hic', 'gex']:
        assert key in batch
        # Assuming batch dimension is the second axis (batch_size == 3)
        assert batch[key].shape[1] == 3


def test_n_bins_extraction(tmp_path: Path):
    # Create a chromsizes file with one chromosome.
    chromsizes_file = tmp_path / "chrom.sizes"
    chromsizes_file.write_text("chr1\t1000\n")
    
    # Create a consensus BED file with one region: chr1 100 150 (width = 50).
    consensus = pd.DataFrame({
        0: ["chr1"],
        1: [100],
        2: [150]
    })
    # Also create a 'region' column as used by import_bigwigs.
    consensus["region"] = consensus[0] + ":" + consensus[1].astype(str) + "-" + consensus[2].astype(str)
    consensus_file = tmp_path / "consensus.bed"
    consensus_file.write_text(consensus.to_csv(sep="\t", header=False, index=False))
    
    # Create a dummy BigWig file using pybigtools.
    bigwigs_dir = tmp_path / "bigwigs"
    bigwigs_dir.mkdir()
    bigwig_file = bigwigs_dir / "test.bw"
    import pybigtools
    bw = pybigtools.open(str(bigwig_file), mode="w")
    # Write a constant signal (5.0) over chr1 (length 1000)
    bw.write(chroms={"chr1": 1000}, vals=[("chr1", 0, 1000, 5.0)])
    bw.close()
    
    # Define an HDF5 backing path.
    backed_path = tmp_path / "data.h5"
    
    # Set target_region_width = 50, target mode = "raw" (which uses bw.values and the bins parameter)
    target_region_width = 50
    target = "raw"
    
    # --- Test with n_bins set to 10 ---
    adata_bins = import_bigwigs(
        bigwigs_folder=bigwigs_dir,
        regions_file=consensus_file,
        backed_path=str(backed_path),
        target_region_width=target_region_width,
        target=target,
        chromsizes_file=str(chromsizes_file),
        n_bins=10
    )
    # Check that X has shape (n_obs, n_var, 10)
    assert adata_bins.X.shape[2] == 10, f"Expected 10 bins, got {adata_bins.X.shape[2]}"
    
    # --- Test with n_bins as None ---
    backed_path2 = tmp_path / "data2.h5"
    adata_full = import_bigwigs(
        bigwigs_folder=bigwigs_dir,
        regions_file=consensus_file,
        backed_path=str(backed_path2),
        target_region_width=target_region_width,
        target=target,
        chromsizes_file=str(chromsizes_file),
        n_bins=None
    )
    # Without binning, the extracted sequence length should equal the target_region_width.
    assert adata_full.X.shape[2] == target_region_width, f"Expected {target_region_width} values, got {adata_full.X.shape[2]}"


def test_meta_module_two_genomes(tmp_path: Path):
    # Create a chromsizes file.
    chromsizes_file = tmp_path / "chrom.sizes"
    chromsizes_file.write_text("chr1\t1000\n")
    
    # Create two FASTA files with distinct sequences.
    fasta1 = tmp_path / "genome1.fa"
    fasta2 = tmp_path / "genome2.fa"
    # Genome1: sequence of "AT" repeated.
    seq1 = "AT" * 500  
    # Genome2: sequence of "CG" repeated.
    seq2 = "CG" * 500  
    fasta1.write_text(">chr1\n" + seq1 + "\n")
    fasta2.write_text(">chr1\n" + seq2 + "\n")
    
    # Create two Genome objects.
    genome1 = Genome(str(fasta1), chrom_sizes=str(chromsizes_file))
    genome2 = Genome(str(fasta2), chrom_sizes=str(chromsizes_file))
    
    # Create a consensus BED file for a single region: chr1:100-110:+
    consensus = pd.DataFrame({
        0: ["chr1"],
        1: [100],
        2: [110]
    })
    consensus_file = tmp_path / "consensus.bed"
    consensus_file.write_text(consensus.to_csv(sep="\t", header=False, index=False))
    
    # Create a simple BigWig file using pybigtools.
    bigwigs_dir = tmp_path / "bigwigs"
    bigwigs_dir.mkdir()
    bigwig_file = bigwigs_dir / "test.bw"
    import pybigtools
    bw = pybigtools.open(str(bigwig_file), mode="w")
    bw.write(chroms={"chr1": 1000}, vals=[("chr1", 0, 1000, 5.0)])
    bw.close()
    
    # Path for HDF5 backing file.
    backed_path = tmp_path / "chrom_data.h5"
    
    # Use import_bigwigs to populate sequence information.
    adata1 = import_bigwigs(
        bigwigs_folder=bigwigs_dir,
        regions_file=consensus_file,
        backed_path=str(backed_path),
        target_region_width=10,
        chromsizes_file=str(chromsizes_file),
    )
    adata2 = import_bigwigs(
        bigwigs_folder=bigwigs_dir,
        regions_file=consensus_file,
        backed_path=str(backed_path),
        target_region_width=10,
        chromsizes_file=str(chromsizes_file),
    )
    
    # Override obs and var with our known values.
    # We create a var DataFrame with proper "chr", "start", and "end" columns.
    obs = pd.DataFrame(index=["obs0"])
    var = pd.DataFrame({
        "chr": ["chr1"],
        "start": [100],
        "end": [110],
        "split": ["train"]
    }, index=["chr1:100-110:+"])
    adata1.obs = obs
    adata1.var = var
    adata2.obs = obs.copy()
    adata2.var = var.copy()
    
    # Create a MetaAnnDataModule with the two datasets and their genomes.
    meta_module = MetaAnnDataModule(
        adatas=[adata1, adata2],
        genomes=[genome1, genome2],
        data_sources={'y': 'X'},
        in_memory=True,
        random_reverse_complement=False,
        max_stochastic_shift=0,
        deterministic_shift=False,
        shuffle_obs=False,
        shuffle=False,
        batch_size=1,
        epoch_size=2
    )
    meta_module.setup("fit")
    loader = meta_module.train_dataloader
    
    # Get all samples from the meta dataset.
    samples = [meta_module.train_dataset[i] for i in range(len(meta_module.train_dataset))]
    assert len(samples) == 2, "Expected 2 samples in meta dataset"
    
    # Decode the one-hot encoded sequence from each sample.
    decoded_sequences = [hot_encoding_to_sequence(s["sequence"]) for s in samples]
    
    # The expected substring is from positions 100 to 110 (10 bp) from each genome.
    expected_seq1 = seq1[100:110].upper()  # For genome1
    expected_seq2 = seq2[100:110].upper()  # For genome2
    
    # Check that one sample's decoded sequence matches expected_seq1 and the other matches expected_seq2.
    assert expected_seq1 in decoded_sequences, f"Expected {expected_seq1} not found in {decoded_sequences}"
    assert expected_seq2 in decoded_sequences, f"Expected {expected_seq2} not found in {decoded_sequences}"
