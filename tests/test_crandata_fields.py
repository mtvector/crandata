import os
from collections import defaultdict
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import pybigtools  # Use pybigtools instead of pyBigWig

from crandata.crandata import CrAnData
from crandata.chrom_io import import_bigwigs, add_contact_strengths_to_varp
from crandata._genome import Genome
from crandata._anndatamodule import MetaAnnDataModule
from crandata._dataloader import AnnDataLoader

# -----------------------------------------------------------------------------
# Test 1: CrAnData fields and properties
# -----------------------------------------------------------------------------

def test_yanndata_fields(tmp_path: Path):
    # Create dummy data for all fields
    X = xr.DataArray(np.arange(20).reshape(4, 5), dims=["obs", "var"])
    obs = pd.DataFrame({"a": list("ABCD")}, index=["obs1", "obs2", "obs3", "obs4"])
    var = pd.DataFrame({"b": list("VWXYZ")[:5]}, index=["var1", "var2", "var3", "var4", "var5"])
    obsm = {"embedding": xr.DataArray(np.random.rand(4, 2), dims=["obs", "dim"])}
    varm = {"feature": xr.DataArray(np.random.rand(5, 3), dims=["var", "dim"])}
    layers = {"layer1": X.copy()}
    varp = {"contacts": xr.DataArray(np.random.rand(5, 5), dims=["var_0", "var_1"])}
    obsp = {"adj": xr.DataArray(np.random.rand(4, 4), dims=["obs_0", "obs_1"])}
    
    ydata = CrAnData(X, obs=obs, var=var, uns={"extra": "test"},
                     obsm=obsm, varm=varm, layers=layers, varp=varp, obsp=obsp)
    
    # Test that getters return the expected content
    assert np.array_equal(ydata.X.data, X.data)
    pd.testing.assert_frame_equal(ydata.obs, obs)
    pd.testing.assert_frame_equal(ydata.var, var)
    assert "embedding" in ydata.obsm
    assert "feature" in ydata.varm
    assert "layer1" in ydata.layers
    assert "contacts" in ydata.varp
    assert "adj" in ydata.obsp
    # Test properties for shape and names
    assert ydata.shape == X.shape
    assert list(ydata.obs_names) == list(obs.index)
    assert list(ydata.var_names) == list(var.index)

# -----------------------------------------------------------------------------
# Test 2: obs loading after HDF5 save/load
# -----------------------------------------------------------------------------

def test_obs_loaded_correctly(tmp_path: Path):
    # Create a simple CrAnData object and save it to HDF5.
    X = xr.DataArray(np.arange(12).reshape(3, 4), dims=["obs", "var"])
    obs = pd.DataFrame({"col": [1, 2, 3]}, index=["o1", "o2", "o3"])
    var = pd.DataFrame({"col": [10, 20, 30, 40]}, index=["v1", "v2", "v3", "v4"])
    ydata = CrAnData(X, obs=obs, var=var)
    h5_path = tmp_path / "test_adata.h5"
    ydata.to_h5(str(h5_path))
    ydata_loaded = CrAnData.from_h5(str(h5_path))
    pd.testing.assert_frame_equal(ydata_loaded.obs, obs)
    pd.testing.assert_frame_equal(ydata_loaded.var, var)

# -----------------------------------------------------------------------------
# Test 3: Batches composition and size from AnnDataLoader
# -----------------------------------------------------------------------------

def test_batches_composition():
    # Create a small dummy CrAnData object
    X = xr.DataArray(np.random.rand(6, 10), dims=["obs", "var"])
    obs = pd.DataFrame({"col": np.arange(6)}, index=[f"obs{i}" for i in range(6)])
    var = pd.DataFrame({"col": np.arange(10)}, index=[f"var{j}" for j in range(10)])
    ydata = CrAnData(X, obs=obs, var=var)
    
    # Create a dummy dataset that mimics the __getitem__ behavior of AnnDataset.
    class DummyDataset:
        def __init__(self, ydata):
            self.ydata = ydata
            self.augmented_probs = np.ones(6)
            self.index_manager = type("IM", (), {"augmented_indices": list(range(6))})
        def __getitem__(self, idx):
            # Return a dict with 'sequence' and 'y' keys
            seq = self.ydata.X.isel(obs=idx).data  # shape (var,)
            return {"sequence": seq, "y": seq}
        def __len__(self):
            return 6
    
    dataset = DummyDataset(ydata)
    loader = AnnDataLoader(dataset, batch_size=2, shuffle=False,
                           drop_remainder=False, epoch_size=6, stage="test",
                           shuffle_obs=False)
    batch = next(iter(loader.data))
    # Verify that the keys exist and batch dimension (second axis) equals batch_size.
    for key in ['sequence', 'y']:
        assert key in batch
        assert batch[key].shape[1] == 2

# -----------------------------------------------------------------------------
# Test 4: Backed files and lazy loading via CrAnData.from_h5
# -----------------------------------------------------------------------------

def test_lazy_loading(tmp_path: Path):
    X = xr.DataArray(np.random.rand(50, 20), dims=["obs", "var"])
    ydata = CrAnData(X)
    h5_path = tmp_path / "lazy.h5"
    ydata.to_h5(str(h5_path))
    ydata_loaded = CrAnData.from_h5(str(h5_path), backed=["X"])
    lazy_obj = ydata_loaded._X.attrs.get("_lazy_obj")
    assert lazy_obj is not None, "Lazy object not found in attributes"
    assert lazy_obj.__class__.__name__ == "LazyH5Array"

# -----------------------------------------------------------------------------
# Test 5: DNA sequence retrieval and shifting correctness
# -----------------------------------------------------------------------------

def test_dna_sequence_retrieval_and_shift(tmp_path: Path):
    # Create a dummy FASTA file with a known sequence.
    fasta_file = tmp_path / "chr1.fa"
    # Build a sequence by repeating "ACGT" to length 1000.
    seq = ("ACGT" * 250)[:1000]
    fasta_file.write_text(">chr1\n" + seq + "\n")
    
    # Create a dummy chromsizes file.
    chromsizes_file = tmp_path / "chrom.sizes"
    chromsizes_file.write_text("chr1\t1000\n")
    
    # Build a Genome object.
    genome = Genome(str(fasta_file), chrom_sizes=str(chromsizes_file))
    
    # Import the SequenceLoader from _dataset.
    from crandata._dataset import SequenceLoader
    # Use a simple region: from 100 to 110 on chr1.
    regions = ["chr1:100-110:+"]
    loader = SequenceLoader(genome, in_memory=True, always_reverse_complement=False,
                              deterministic_shift=False, max_stochastic_shift=5, regions=regions)
    # Retrieve the sequence with no additional shift.
    retrieved_seq = loader.get_sequence("chr1:100-110:+", shift=0)
    expected_seq = seq[100:110]
    assert retrieved_seq == expected_seq

    # Test with a shift of 2: the expected sequence shifts accordingly.
    retrieved_seq_shift = loader.get_sequence("chr1:100-110:+", shift=2)
    expected_seq_shift = seq[102:112]
    assert retrieved_seq_shift == expected_seq_shift

# -----------------------------------------------------------------------------
# Test 6: MetaAnnDataModule loads the correct genomic DNA per row
# -----------------------------------------------------------------------------

def test_meta_module_genomic_dna(tmp_path: Path):
    # Create a dummy FASTA with known sequence.
    fasta_file = tmp_path / "chr1.fa"
    seq = ("ACGT" * 250)[:1000]
    fasta_file.write_text(">chr1\n" + seq + "\n")
    
    chromsizes_file = tmp_path / "chrom.sizes"
    chromsizes_file.write_text("chr1\t1000\n")
    genome = Genome(str(fasta_file), chrom_sizes=str(chromsizes_file))
    
    # Create a consensus BED file for three regions.
    consensus = pd.DataFrame({
        0: ["chr1"] * 3,
        1: [100, 200, 300],
        2: [110, 210, 310]
    })
    consensus_file = tmp_path / "consensus.bed"
    consensus_file.write_text(consensus.to_csv(sep="\t", header=False, index=False))
    
    # Create a simple BigWig file using pybigtools.
    bigwigs_dir = tmp_path / "bigwigs"
    bigwigs_dir.mkdir()
    bigwig_file = bigwigs_dir / "test.bw"
    # Open a BigWig file for writing.
    bw = pybigtools.open(str(bigwig_file), mode="w")
    # Write the file using a dictionary for chromosome lengths and an iterable of tuples.
    bw.write(chroms={"chr1": 1000}, vals=[("chr1", 0, 1000, 5.0)])
    bw.close()
    
    backed_path = tmp_path / "chrom_data.h5"
    adata = import_bigwigs(
        bigwigs_folder=bigwigs_dir,
        regions_file=consensus_file,
        backed_path=str(backed_path),
        target_region_width=10,
        chromsizes_file=str(chromsizes_file),
    )
    adata.var["split"] = "train"
    adata.obsm["dummy"] = xr.DataArray(np.random.rand(adata.obs.shape[0], 5),
                                        dims=["types", "genes"])
    # Create two copies of adata.
    adata1 = adata.copy()
    adata2 = adata.copy()
    meta_module = MetaAnnDataModule(
        adatas=[adata1, adata2],
        genomes=[genome, genome],
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
    batch = next(iter(loader.data))
    # Decode the one-hot encoded "sequence" to a DNA string.
    from crandata.utils import hot_encoding_to_sequence
    sample_seq = batch["sequence"][:, 0]
    # Convert sample_seq to a NumPy array (if it's a torch tensor)
    sample_seq_np = sample_seq.cpu().numpy() if hasattr(sample_seq, "cpu") else np.asarray(sample_seq)
    decoded_seq = hot_encoding_to_sequence(sample_seq_np)
    # The first consensus region is from 100 to 110.
    expected_seq = seq[100:110]
    assert decoded_seq == expected_seq

# -----------------------------------------------------------------------------
# Test 7: Verify that obs dimensions are shuffled correctly in the batch
# -----------------------------------------------------------------------------


def test_obs_shuffling(monkeypatch):
    import torch
    # Create a dummy sample with distinct observation rows.
    dummy_sample = {
        "sequence": np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]]),  # shape (3, 3)
        "y": np.array([[10, 11, 12],
                       [13, 14, 15],
                       [16, 17, 18]])
    }
    # Force a fixed permutation using torch.randperm.
    fixed_perm = torch.tensor([2, 0, 1])
    monkeypatch.setattr(torch, "randperm", lambda n: fixed_perm)
    
    from crandata._dataloader import AnnDataLoader
    # Create a dummy dataset that returns dummy_sample.
    class DummyDataset:
        def __getitem__(self, idx):
            return dummy_sample
        def __len__(self):
            return 1
    dataset = DummyDataset()
    loader = AnnDataLoader(dataset, batch_size=1, shuffle_obs=True)
    # The collate function expects a list of samples.
    batch = loader.batch_collate_fn([dummy_sample, dummy_sample])
    # Each key is stacked along axis=1.
    # Since we have two samples, the expected shape is (3, 2, 3) with the same permutation applied.
    expected_sequence = np.stack([dummy_sample["sequence"][fixed_perm.numpy()]] * 2, axis=1)
    expected_y = np.stack([dummy_sample["y"][fixed_perm.numpy()]] * 2, axis=1)
    np.testing.assert_array_equal(batch["sequence"].cpu().numpy(), expected_sequence)
    np.testing.assert_array_equal(batch["y"].cpu().numpy(), expected_y)
