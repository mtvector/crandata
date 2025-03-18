"""_dataloader.py – DataLoader for batching, shuffling, and one‐hot encoding
of CrAnData-based AnnDataset objects using the new CrAnData structure.

This version no longer uses LazyData, and implements its own helper (_reindex_array)
to align the observation dimension when needed.
"""

from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import xarray as xr
import h5py

# Import the updated dataset classes that wrap a CrAnData object.
from ._dataset import AnnDataset, MetaAnnDataset

# ------------------------------------------------------------------------------
def _reindex_array(array: np.ndarray, local_obs: np.ndarray, global_obs: np.ndarray) -> np.ndarray:
    """
    Given a NumPy array whose first dimension corresponds to local observations,
    create a new array with first dimension equal to len(global_obs). For observations
    present in local_obs, copy the values; for missing ones, fill with NaN.
    """
    new_shape = (len(global_obs),) + array.shape[1:]
    new_array = np.full(new_shape, np.nan, dtype=array.dtype)
    for i, obs in enumerate(local_obs):
        idx = np.where(global_obs == obs)[0]
        if idx.size:
            new_array[idx[0]] = array[i]
    return new_array

# ------------------------------------------------------------------------------
def _shuffle_obs_in_sample(sample: dict) -> dict:
    """
    Given a sample dictionary (mapping keys to NumPy arrays), determine a permutation 
    over the observation dimension (assumed to be the first dimension) and apply it.
    """
    if "sequence" in sample:
        n_obs = sample["sequence"].shape[0]
    else:
        n_obs = None
        for val in sample.values():
            arr = np.asarray(val)
            if arr.ndim > 0:
                n_obs = arr.shape[0]
                break
    if n_obs is None:
        return sample
    perm = np.random.permutation(n_obs)
    new_sample = {}
    for key, val in sample.items():
        arr = np.asarray(val)
        if arr.ndim > 0 and arr.shape[0] == n_obs:
            new_sample[key] = arr[perm]
        else:
            new_sample[key] = arr
    return new_sample

# ------------------------------------------------------------------------------
class WeightedRegionSampler(Sampler):
    """
    Sampler that randomly samples augmented region indices from an AnnDataset in 
    proportion to their (nonzero) weights.
    """
    def __init__(self, dataset: AnnDataset, epoch_size: int = 100_000):
        super().__init__(data_source=dataset)
        self.dataset = dataset
        self.epoch_size = epoch_size
        p = dataset.augmented_probs
        s = p.sum()
        if s <= 0:
            raise ValueError("All sample probabilities are zero, cannot sample.")
        self.probs = p / s

    def __iter__(self):
        n = len(self.dataset.index_manager.augmented_indices)
        for _ in range(self.epoch_size):
            yield np.random.choice(n, p=self.probs)

    def __len__(self):
        return self.epoch_size

# ------------------------------------------------------------------------------
class NonShuffleRegionSampler(Sampler):
    """
    Deterministically iterate over all augmented region indices (with nonzero probability)
    exactly once, in ascending order.
    """
    def __init__(self, dataset: AnnDataset):
        super().__init__(data_source=dataset)
        self.dataset = dataset
        p = self.dataset.augmented_probs
        self.nonzero_indices = np.flatnonzero(p > 0.0)
        if len(self.nonzero_indices) == 0:
            raise ValueError("No nonzero probabilities for validation/test stage.")

    def __iter__(self):
        return iter(self.nonzero_indices)

    def __len__(self):
        return len(self.nonzero_indices)

# ------------------------------------------------------------------------------
class MetaSampler(Sampler):
    """
    Sampler for a MetaAnnDataset that yields global indices according to global_probs.
    """
    def __init__(self, meta_dataset: MetaAnnDataset, epoch_size: int = 100_000):
        super().__init__(data_source=meta_dataset)
        self.meta_dataset = meta_dataset
        self.epoch_size = epoch_size
        s = self.meta_dataset.global_probs.sum()
        if not np.isclose(s, 1.0, atol=1e-6):
            raise ValueError(f"Global probabilities do not sum to 1 after normalization. Sum = {s}")

    def __iter__(self):
        n = len(self.meta_dataset)
        p = self.meta_dataset.global_probs
        for _ in range(self.epoch_size):
            yield np.random.choice(n, p=p)

    def __len__(self):
        return self.epoch_size

# ------------------------------------------------------------------------------
class NonShuffleMetaSampler(Sampler):
    """
    Deterministically iterate over all global indices (with nonzero probability)
    exactly once for MetaAnnDataset.
    """
    def __init__(self, meta_dataset: MetaAnnDataset, sort: bool = True):
        super().__init__(data_source=meta_dataset)
        self.meta_dataset = meta_dataset
        p = self.meta_dataset.global_probs
        self.nonzero_global_indices = np.flatnonzero(p > 0)
        if sort:
            self.nonzero_global_indices.sort()

    def __iter__(self):
        return iter(self.nonzero_global_indices)

    def __len__(self):
        return len(self.nonzero_global_indices)

# ------------------------------------------------------------------------------
class GroupedChunkMetaSampler(Sampler):
    """
    Sampler for a MetaAnnDataset that first selects one dataset (file) and then a chunk 
    within that dataset based on probabilities, returning a tuple (dataset index, local index).
    """
    def __init__(self, meta_dataset: MetaAnnDataset, epoch_size: int = 100_000):
        super().__init__(data_source=meta_dataset)
        self.meta_dataset = meta_dataset
        self.epoch_size = epoch_size
        self.file_probs = meta_dataset.file_probs

    def __iter__(self):
        for _ in range(self.epoch_size):
            ds_idx = np.random.choice(len(self.meta_dataset.datasets), p=self.file_probs)
            dataset = self.meta_dataset.datasets[ds_idx]
            chunk_keys = list(dataset.chunk_weights.keys())
            chunk_weights = np.array([dataset.chunk_weights[ch] for ch in chunk_keys])
            if chunk_weights.sum() <= 0:
                chunk_probs = np.ones_like(chunk_weights) / len(chunk_weights)
            else:
                chunk_probs = chunk_weights / chunk_weights.sum()
            chosen_chunk = np.random.choice(chunk_keys, p=chunk_probs)
            local_indices = dataset.chunk_groups[chosen_chunk]
            local_probs = dataset.augmented_probs[local_indices]
            if local_probs.sum() <= 0:
                local_probs = np.ones_like(local_probs)
            else:
                local_probs = local_probs / local_probs.sum()
            chosen_local_idx = np.random.choice(local_indices, p=local_probs)
            yield (ds_idx, chosen_local_idx)

    def __len__(self):
        return self.epoch_size

# ------------------------------------------------------------------------------
class AnnDataLoader:
    """
    PyTorch-like DataLoader for CrAnData-based AnnDataset (or MetaAnnDataset) objects.
    This loader provides batching, shuffling, and (optional) one-hot encoding.
    
    Parameters
    ----------
    dataset
        An instance of AnnDataset or MetaAnnDataset wrapping a CrAnData object.
    batch_size
        Number of samples per batch.
    shuffle
        Whether to shuffle the dataset (if no custom sampler is provided).
    drop_remainder
        If True, drop the last incomplete batch.
    epoch_size
        Number of samples to draw in one epoch (used for weighted sampling).
    stage
        Stage indicator ("train", "val", "test", etc.) to select the appropriate sampler.
    shuffle_obs
        If True, shuffle the observation dimension within each batch.
    
    Example
    -------
    >>> dataset = AnnDataset(...)  # wraps a CrAnData object
    >>> loader = AnnDataLoader(dataset, batch_size=32, shuffle=True, stage="train")
    >>> for batch in loader.data:
    ...     # process the batch
    """
    def __init__(
        self,
        dataset,  # AnnDataset or MetaAnnDataset
        batch_size: int,
        shuffle: bool = False,
        drop_remainder: bool = True,
        epoch_size: int = 100_000,
        stage: str = "train",
        shuffle_obs: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.epoch_size = epoch_size
        self.stage = stage
        self.shuffle_obs = shuffle_obs

        if os.environ.get("KERAS_BACKEND", "") == "torch":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

        self.sampler = None
        # For MetaAnnDataset, choose meta-level sampling.
        if isinstance(dataset, MetaAnnDataset):
            if self.stage == "train":
                self.sampler = GroupedChunkMetaSampler(dataset, epoch_size=self.epoch_size)
            else:
                self.sampler = NonShuffleMetaSampler(dataset, sort=True)
        else:
            # For a single AnnDataset with augmented probabilities.
            if getattr(dataset, "augmented_probs", None) is not None:
                if self.stage == "train":
                    self.sampler = WeightedRegionSampler(dataset, epoch_size=self.epoch_size)
                else:
                    self.sampler = NonShuffleRegionSampler(dataset)
            else:
                if self.shuffle and hasattr(self.dataset, "shuffle"):
                    self.dataset.shuffle = True

    def batch_collate_fn(self, batch):
        """
        Collate function to combine a list of sample dictionaries.
        For numeric keys that are not global, convert them into torch tensors by stacking;
        for global keys (e.g. hi-C), simply take the value from the first sample.
        For non-numeric keys, return a list of values.
        Additionally, for keys with a global observation ordering, reindex if necessary.
        """
        collated = {}
        global_keys = {"hic"}  # keys that should not be stacked (assumed to be global)
        global_obs = None
        local_obs = None
        # For MetaAnnDataset, use the global observation names from the first dataset.
        if isinstance(self.dataset, MetaAnnDataset) and len(self.dataset.datasets) > 0:
            global_obs = np.array(self.dataset.datasets[0].adata.meta_obs_names)
            local_obs = np.array(self.dataset.datasets[0].adata["obs/index"])
                
        for key in batch[0]:
            collected = []
            for sample in batch:
                val = sample[key]
                # If val is a sparse array, densify it.
                if hasattr(val, "todense"):
                    arr = np.array(val.todense())
                else:
                    arr = np.array(val)
                if arr.ndim == 0:
                    arr = np.expand_dims(arr, 0)
                if global_obs is not None and key not in ["sequence"] and arr.shape[0] != len(global_obs):
                    if np.issubdtype(arr.dtype, np.number):
                        arr = _reindex_array(arr, local_obs, global_obs)
                collected.append(arr)
            if key in global_keys:
                # For global keys, assume all samples are identical; take the first.
                collated[key] = torch.as_tensor(collected[0], dtype=torch.float32)
            elif np.issubdtype(np.array(collected[0]).dtype, np.number):
                collated[key] = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in collected], dim=0)
            else:
                collated[key] = collected
    
        if self.shuffle_obs and global_obs is not None:
            obs_expected = len(global_obs)
            perm = torch.randperm(obs_expected)
            for key, val in collated.items():
                if isinstance(val, torch.Tensor):
                    # Search for the axis that has size equal to the expected number of observations.
                    obs_axis = None
                    for axis, dim_size in enumerate(val.shape):
                        if dim_size == obs_expected:
                            obs_axis = axis
                            break
                    if obs_axis is not None:
                        # Bring the observation axis to the front.
                        if obs_axis != 0:
                            permuted = val.transpose(0, obs_axis)
                        else:
                            permuted = val
                        # Apply the permutation along the first axis.
                        permuted = permuted[perm]
                        # Transpose back if needed.
                        if obs_axis != 0:
                            permuted = permuted.transpose(0, obs_axis)
                        collated[key] = permuted
        if self.device is not None:
            for key, val in collated.items():
                if isinstance(val, torch.Tensor):
                    collated[key] = val.to(self.device)
        return collated

    def _create_dataset(self):
        """
        Creates the underlying data generator. A torch DataLoader is returned.
        """
        if os.environ.get("KERAS_BACKEND", "") == "torch":
            if self.sampler is not None:
                return DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    sampler=self.sampler,
                    drop_last=self.drop_remainder,
                    num_workers=0,
                    collate_fn=self.batch_collate_fn,
                )
            else:
                return DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    drop_last=self.drop_remainder,
                    num_workers=0,
                    collate_fn=self.batch_collate_fn,
                )
        else:
            raise NotImplementedError("Only PyTorch backend is implemented in this dataloader.")

    @property
    def data(self):
        return self._create_dataset()

    def __len__(self):
        if self.sampler is not None:
            return (self.epoch_size + self.batch_size - 1) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __repr__(self):
        return (
            f"AnnDataLoader(dataset={self.dataset}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, drop_remainder={self.drop_remainder}, "
            f"shuffle_obs={self.shuffle_obs})"
        )
