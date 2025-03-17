"""_dataloader.py – Dataloader for batching, shuffling, and one-hot encoding of CrAnData-based AnnDataset objects."""

from __future__ import annotations
import os
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import xarray as xr
import h5py

try:
    import sparse
except ImportError:
    sparse = None

# Optionally try to use xbatcher for batching.
try:
    import xbatcher
except ImportError:
    xbatcher = None

# Instead of the old LazyData and reindex_obs_array, we define a simple reindexing helper.
def _reindex_obs_array(arr: np.ndarray, local_obs: np.ndarray, global_obs: np.ndarray) -> np.ndarray:
    """
    Reindex the observation axis of an array so that its first dimension matches the global observations.
    Missing observations are filled with NaN.
    """
    new_shape = (len(global_obs),) + arr.shape[1:]
    new_arr = np.full(new_shape, np.nan, dtype=arr.dtype)
    local_to_global = {obs: i for i, obs in enumerate(global_obs)}
    for i, obs in enumerate(local_obs):
        if obs in local_to_global:
            new_arr[local_to_global[obs]] = arr[i]
    return new_arr

from ._dataset import AnnDataset, MetaAnnDataset

os.environ["KERAS_BACKEND"] = "torch"

def _shuffle_obs_in_sample(sample: dict) -> dict:
    # Determine number of observations from the "sequence" key or from any array.
    n_obs = None
    if "sequence" in sample:
        n_obs = sample["sequence"].shape[0]
    else:
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

# --- Sampler classes remain largely unchanged ---
class WeightedRegionSampler(Sampler):
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

class NonShuffleRegionSampler(Sampler):
    def __init__(self, dataset: AnnDataset):
        super().__init__(data_source=dataset)
        self.dataset = dataset
        p = self.dataset.augmented_probs
        self.nonzero_indices = np.flatnonzero(p > 0.0)
        if len(self.nonzero_indices) == 0:
            raise ValueError("No nonzero probabilities for val/test stage.")

    def __iter__(self):
        return iter(self.nonzero_indices)

    def __len__(self):
        return len(self.nonzero_indices)

class MetaSampler(Sampler):
    def __init__(self, meta_dataset: MetaAnnDataset, epoch_size: int = 100_000):
        super().__init__(data_source=meta_dataset)
        self.meta_dataset = meta_dataset
        self.epoch_size = epoch_size
        s = self.meta_dataset.global_probs.sum()
        if not np.isclose(s, 1.0, atol=1e-6):
            raise ValueError("Global probabilities do not sum to 1 after normalization. Sum = {}".format(s))

    def __iter__(self):
        n = len(self.meta_dataset)
        p = self.meta_dataset.global_probs
        for _ in range(self.epoch_size):
            yield np.random.choice(n, p=p)

    def __len__(self):
        return self.epoch_size

class NonShuffleMetaSampler(Sampler):
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

class GroupedChunkMetaSampler(Sampler):
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

# --- Dataloader class ---
class AnnDataLoader:
    def __init__(
        self,
        dataset,  # either AnnDataset or MetaAnnDataset
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampler = None

        if isinstance(dataset, MetaAnnDataset):
            if self.stage == "train":
                self.sampler = GroupedChunkMetaSampler(dataset, epoch_size=self.epoch_size)
            else:
                self.sampler = NonShuffleMetaSampler(dataset, sort=True)
        else:
            if getattr(dataset, "augmented_probs", None) is not None:
                if self.stage == "train":
                    self.sampler = WeightedRegionSampler(dataset, epoch_size=self.epoch_size)
                else:
                    self.sampler = NonShuffleRegionSampler(dataset)
            else:
                if self.shuffle and hasattr(self.dataset, "shuffle"):
                    self.dataset.shuffle = True

    def batch_collate_fn(self, batch: list[dict]) -> dict:
        collated = {}
        # Determine global observation names from the first dataset.
        if hasattr(self.dataset, "datasets") and len(self.dataset.datasets) > 0:
            global_obs = np.array(self.dataset.datasets[0].adata.meta_obs_names)
            local_obs = np.array(self.dataset.datasets[0].adata.meta_obs_names)
        else:
            # If not a meta dataset, assume the AnnDataset has a property for obs names.
            global_obs = np.array(self.dataset.adata.meta_obs_names)
            local_obs = global_obs
        for key in batch[0]:
            tensors = []
            for sample in batch:
                val = sample[key]
                # If backed by a sparse array, densify.
                if hasattr(val, "data") and hasattr(val.data, "todense"):
                    arr = np.array(val.data.todense())
                else:
                    arr = np.array(val)
                if arr.ndim == 0:
                    arr = np.expand_dims(arr, 0)
                # For keys other than "sequence", if the obs dimension is smaller than global_obs, reindex.
                if global_obs is not None and key != "sequence" and arr.shape[0] != len(global_obs):
                    arr = _reindex_obs_array(arr, local_obs, global_obs)
                tensors.append(torch.as_tensor(arr, dtype=torch.float32))
            # Stack tensors along a new batch dimension.
            stacked = torch.stack(tensors, dim=1)
            collated[key] = stacked
        # Optionally shuffle observations (first dimension) if desired.
        if self.shuffle_obs and global_obs is not None:
            perm = torch.randperm(len(global_obs))
            for key in collated:
                collated[key] = collated[key][perm]
        for key in collated:
            collated[key] = collated[key].to(self.device)
        return collated

    def _create_dataset(self):
        # If xbatcher is available, use it; otherwise fall back to torch DataLoader.
        if xbatcher is not None:
            return xbatcher.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=self.sampler,
                drop_last=self.drop_remainder,
                collate_fn=self.batch_collate_fn,
            )
        else:
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
            f"shuffle={self.shuffle}, drop_remainder={self.drop_remainder}, shuffle_obs={self.shuffle_obs})"
        )
