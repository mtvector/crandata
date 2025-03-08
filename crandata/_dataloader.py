"""_dataloader.py – Dataloader for batching, shuffling, and one-hot encoding of CrAnData-based AnnDataset objects."""

from __future__ import annotations
import os
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import xarray as xr

try:
    import sparse
except ImportError:
    sparse = None

# Import the updated AnnDataset and MetaAnnDataset that now wrap a CrAnData object.
from ._dataset import AnnDataset, MetaAnnDataset

os.environ["KERAS_BACKEND"] = "torch" #TODO TF stuff, nmj
# if os.environ.get("KERAS_BACKEND", "") == "torch":
#     import torch
#     from torch.utils.data import DataLoader, Sampler
# else:
#     import tensorflow as tf

def _shuffle_obs_in_sample(sample: dict) -> dict:
    """
    Given a sample dictionary (mapping keys to NumPy arrays),
    determine a permutation from 0 to N-1 based on one key (here, we assume the
    observation dimension is the first dimension and that at least one key has that dimension).
    Then, for every key whose first dimension matches that length, apply the permutation.
    """
    # Here we assume that the key "sequence" exists and its first dimension is N.
    # You may adjust this if your sample dictionaries use another key.
    if "sequence" in sample:
        n_obs = sample["sequence"].shape[0]
    else:
        # Alternatively, if no "sequence" key is present, use the first key that has ndim>=1.
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

class WeightedRegionSampler(Sampler):
    """
    Sampler that randomly samples augmented region indices from a CrAnData-based AnnDataset
    in proportion to their (nonzero) weights (augmented_probs). Used for training.
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


class NonShuffleRegionSampler(Sampler):
    """
    Sampler that deterministically iterates over all augmented region indices (with nonzero probability)
    exactly once in ascending order. Typically used for validation or test stages.
    """
    def __init__(self, dataset: AnnDataset):
        super().__init__(data_source=dataset)
        self.dataset = dataset
        # Filter out indices with zero probability.
        p = self.dataset.augmented_probs
        self.nonzero_indices = np.flatnonzero(p > 0.0)
        if len(self.nonzero_indices) == 0:
            raise ValueError("No nonzero probabilities for val/test stage.")

    def __iter__(self):
        return iter(self.nonzero_indices)

    def __len__(self):
        return len(self.nonzero_indices)


class MetaSampler(Sampler):
    """
    Sampler for a MetaAnnDataset that yields global indices according to global_probs.
    Used primarily during training to sample across multiple CrAnData-based AnnDataset objects.
    """
    def __init__(self, meta_dataset: MetaAnnDataset, epoch_size: int = 100_000):
        super().__init__(data_source=meta_dataset)
        self.meta_dataset = meta_dataset
        self.epoch_size = epoch_size
        s = self.meta_dataset.global_probs.sum()
        if not np.isclose(s, 1.0, atol=1e-6):
            raise ValueError(
                "Global probabilities do not sum to 1 after normalization. Sum = {}".format(s)
            )

    def __iter__(self):
        n = len(self.meta_dataset)
        p = self.meta_dataset.global_probs
        for _ in range(self.epoch_size):
            yield np.random.choice(n, p=p)

    def __len__(self):
        return self.epoch_size


class NonShuffleMetaSampler(Sampler):
    """
    Sampler for MetaAnnDataset that enumerates all global indices (with nonzero probability)
    exactly once in a deterministic order. Typically used for validation or testing.
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


class AnnDataLoader:
    """
    Pytorch-like DataLoader for CrAnData-based AnnDataset (or MetaAnnDataset) objects.
    Provides batching, shuffling, and one-hot encoding for genomic sequences and additional data.

    Parameters
    ----------
    dataset
        An instance of AnnDataset or MetaAnnDataset that wraps a CrAnData object.
    batch_size
        Number of samples per batch.
    shuffle
        Whether to shuffle the dataset.
    drop_remainder
        If True, drops the last incomplete batch.
    epoch_size
        Number of samples to draw in one epoch (for weighted sampling).
    stage
        Stage indicator ("train", "val", "test", etc.) used to select the appropriate sampler.

    Example
    -------
    >>> dataset = AnnDataset(...)  # CrAnData-based dataset instance
    >>> loader = AnnDataLoader(dataset, batch_size=32, shuffle=True, stage="train")
    >>> for batch in loader.data:
    ...     # Process the batch
    """
    def __init__(
        self,
        dataset,  # can be AnnDataset or MetaAnnDataset
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
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

        self.sampler = None

        # If the dataset is a MetaAnnDataset, use meta-level sampling
        if isinstance(dataset, MetaAnnDataset):
            if self.stage == "train":
                self.sampler = MetaSampler(dataset, epoch_size=self.epoch_size)
            else:
                self.sampler = NonShuffleMetaSampler(dataset, sort=True)
        else:
            # For a single AnnDataset with augmented probabilities
            if getattr(dataset, "augmented_probs", None) is not None:
                if self.stage == "train":
                    self.sampler = WeightedRegionSampler(dataset, epoch_size=self.epoch_size)
                else:
                    self.sampler = NonShuffleRegionSampler(dataset)
            else:
                # Fallback: use uniform shuffling if available
                if self.shuffle and hasattr(self.dataset, "shuffle"):
                    self.dataset.shuffle = True

    def _collate_fn(self, batch):
        x = defaultdict(list)
        # First, convert each sample’s keys to dense tensors.
        for sample_dict in batch:
            for key, val in sample_dict.items():
                if isinstance(val, xr.DataArray):
                    try:
                        underlying = val.variable.data
                    except Exception:
                        underlying = None
                    if sparse is not None and underlying is not None and isinstance(underlying, sparse.COO):
                        # Manually densify if underlying data is sparse.
                        arr = np.asarray(underlying.todense())
                    else:
                        arr = np.asarray(val)
                else:
                    arr = np.asarray(val)
                # If the resulting array is scalar, expand dims.
                if arr.ndim == 0:
                    arr = np.expand_dims(arr, 0)
                x[key].append(torch.as_tensor(arr, dtype=torch.float32))
        # Now stack the list of tensors along a new batch dimension.
        # We assume each sample's arrays are of shape [n_obs, ...]. After stacking,
        # each key becomes a tensor of shape [n_obs, batch, ...].
        for key in x:
            x[key] = torch.stack(x[key], dim=1)
            # Now optionally, shuffle the obs dimension (axis 0) consistently across all keys.
            if self.shuffle_obs:
                perm = torch.randperm(x[key].shape[0])
                x[key] = x[key][perm]
            if self.device is not None:
                x[key] = x[key].to(self.device)
        return x

    def _create_dataset(self):
        # ... (same as before, using self.sampler if available) ...
        from torch.utils.data import DataLoader
        if os.environ.get("KERAS_BACKEND", "") == "torch":
            if self.sampler is not None:
                return DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    sampler=self.sampler,
                    drop_last=self.drop_remainder,
                    num_workers=0,
                    collate_fn=self._collate_fn,
                )
            else:
                return DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    drop_last=self.drop_remainder,
                    num_workers=0,
                    collate_fn=self._collate_fn,
                )
        elif os.environ.get("KERAS_BACKEND", "") == "tensorflow": #Someone who knows tf will have to deal with this
            ds = tf.data.Dataset.from_generator(
                self.dataset,
                output_signature=(
                    tf.TensorSpec(shape=(self.dataset.seq_len, 4), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.dataset.num_outputs,), dtype=tf.float32),
                ),
            )
            ds = (
                ds.batch(self.batch_size, drop_remainder=self.drop_remainder)
                  .repeat()
                  .prefetch(tf.data.AUTOTUNE)
            )
            return ds

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
