import numpy as np
import torch
import h5py
import xbatcher
from torch.utils.data import IterableDataset


def _reindex_array(array, local_obs:, global_obs):
    """
    Given an array whose first dimension corresponds to local observations,
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


# =============================================================================
# New dataloader implementing a datapipe chain.
# =============================================================================
class CrAnDataPipeLoader(IterableDataset):
    """
    A dataloader that multiplexes batches from a CrAnDataModule (or MetaCrAnDataModule)
    based on the desired state ("train", "val", "test", "predict"). It aggregates
    a number of raw samples (each a dict) via a collate function and (optionally)
    applies a single permutation to designated dimensions (e.g. the observation axis).
    
    Parameters:
      dataset: Either a single CrAnDataModule instance or a MetaCrAnDataModule instance.
      state: String indicating which state to sample ("train", "val", "test", "predict").
      collate_fn: Function to aggregate a list of raw samples into a single batch.
      outer_batch_size: How many raw samples to collect before collating.
      shuffle_obs: If True, shuffle the observation dimension (applied uniformly).
      device: Torch device to move tensors to (or None).
      epoch_size: Total number of iterations in one epoch (optional).
    """
    def __init__(self, dataset, state="train", collate_fn=None, outer_batch_size=1, 
                 shuffle_obs=False, device=None, epoch_size=None):
        self.dataset = dataset
        self.state = state
        # Use the default collate if none is provided.
        self.collate_fn = collate_fn or self.default_batch_collate_fn
        self.outer_batch_size = outer_batch_size
        self.shuffle_obs = shuffle_obs
        self.device = device
        self.epoch_size = epoch_size

    def default_batch_collate_fn(self, batch):
        """
        Collate a list of samples (each a dict mapping keys to NumPy arrays)
        into a single dict of tensors. For keys whose arrays have an observation dimension,
        the same random permutation is applied.
        
        This implementation is adapted from your provided legacy collate function.
        """
        collated = {}
        
        # (Optional) Determine global observation names from metadata if available.
        global_obs = None
        local_obs = None
        # If samples include metadata (e.g. 'meta_obs_names'), use that:
        if 'meta_obs_names' in batch[0]:
            global_obs = np.array(batch[0]['meta_obs_names'])
            local_obs = np.array(batch[0].get('obs_names', []))
        
        for key in batch[0]:
            # If every sample for this key is a LazyData–like object, group them
            # by their underlying lazy object so that file I/O can be batched.
            if all(hasattr(sample[key], 'lazy_obj') for sample in batch):
                groups = {}
                for i, sample in enumerate(batch):
                    ld = sample[key]
                    group_id = id(ld.lazy_obj)
                    groups.setdefault(group_id, []).append((i, ld.key, ld))
                results = [None] * len(batch)
                for group in groups.values():
                    lazy_obj = group[0][2].lazy_obj
                    keys = [item[1] for item in group]
                    with h5py.File(lazy_obj.filename, "r") as f:
                        dset = f[lazy_obj.dataset_name]
                        group_data = [dset[k] for k in keys]
                    for j, (i, _, ld) in enumerate(group):
                        data_item = group_data[j]
                        if getattr(ld, 'global_obs', None) is not None and data_item.shape[0] < len(ld.global_obs) and key not in ["sequence"]:
                            data_item = _reindex_array(data_item, ld.local_obs, ld.global_obs)
                        results[i] = data_item
                collated_value = torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in results], dim=1)
            else:
                tensors = []
                for sample in batch:
                    val = sample[key]
                    # Densify if necessary.
                    if hasattr(val, "data") and hasattr(val.data, "todense"):
                        arr = np.array(val.data.todense())
                    else:
                        arr = np.array(val)
                    if arr.ndim == 0:
                        arr = np.expand_dims(arr, 0)
                    if global_obs is not None and key not in ["sequence"] and arr.shape[0] != len(global_obs):
                        arr = _reindex_array(arr, local_obs, global_obs)
                    tensors.append(torch.as_tensor(arr, dtype=torch.float32))
                collated_value = torch.stack(tensors, dim=1)
            collated[key] = collated_value

        # Now shuffle the observation dimension if requested.
        if self.shuffle_obs:
            obs_dim = None
            if global_obs is not None:
                obs_dim = len(global_obs)
            else:
                for tensor in collated.values():
                    if tensor.ndim > 0:
                        obs_dim = tensor.shape[0]
                        break
            if obs_dim is not None:
                perm = torch.randperm(obs_dim)
                for key in collated:
                    if collated[key].ndim > 0 and collated[key].shape[0] == obs_dim:
                        collated[key] = collated[key][perm]
        if self.device is not None:
            for key in collated:
                collated[key] = collated[key].to(self.device)
                # If needed, adjust dimensions.
                if collated[key].dim() > 1 and collated[key].shape[1] == len(batch) and collated[key].shape[0] != len(batch):
                    collated[key] = collated[key].permute((1, 0) + tuple(range(2, collated[key].dim())))
        return collated

    def __iter__(self):
        """
        Iterates for a number of steps (controlled by epoch_size if provided).
        For a meta dataset (with a .modules attribute), a module is first chosen
        (using its file probability) and then a random batch index is selected from
        that module’s generator for the given state. For a single module, we iterate
        sequentially (wrapping around if necessary).
        """
        batch_list = []
        iterations = 0
        if hasattr(self.dataset, 'modules'):
            # Meta module: self.dataset.modules is a list of CrAnDataModule instances,
            # and self.dataset.file_probs is an array of sampling probabilities.
            while self.epoch_size is None or iterations < self.epoch_size:
                mod_idx = np.random.choice(len(self.dataset.modules), p=self.dataset.file_probs)
                module = self.dataset.modules[mod_idx]
                gen = module._generators.get(self.state)
                if gen is None:
                    raise ValueError(f"Generator for state '{self.state}' not set in module.")
                batch_idx = np.random.randint(0, len(gen))
                sample = module.get_batch(self.state, batch_idx)
                batch_list.append(sample)
                if len(batch_list) == self.outer_batch_size:
                    yield self.collate_fn(batch_list)
                    batch_list = []
                iterations += 1
        else:
            # Single module.
            gen = self.dataset._generators.get(self.state)
            if gen is None:
                raise ValueError(f"Generator for state '{self.state}' not set in module.")
            total = len(gen) if self.epoch_size is None else self.epoch_size
            for i in range(total):
                sample = self.dataset.get_batch(self.state, i % len(gen))
                batch_list.append(sample)
                if len(batch_list) == self.outer_batch_size:
                    yield self.collate_fn(batch_list)
                    batch_list = []
                iterations += 1

    def __len__(self):
        if self.epoch_size is not None:
            return (self.epoch_size + self.outer_batch_size - 1) // self.outer_batch_size
        else:
            if hasattr(self.dataset, 'modules'):
                return sum(len(mod._generators.get(self.state, [])) for mod in self.dataset.modules)
            else:
                return len(self.dataset._generators.get(self.state, []))


# =============================================================================
# New CrAnDataModule and MetaCrAnDataModule with dataloader properties.
# =============================================================================
class CrAnDataModule:
    """
    A module wrapping a CrAnData object (e.g. an xarray–like dataset with genomic data)
    that uses xbatcher for sampling. A DNATransform (if provided) is applied to the
    "sequences" key when batches are retrieved.
    
    Attributes:
      adata: The CrAnData object.
      batch_size: Number of slices to sample along the "var" dimension.
      shuffle_dims: List of dimension names to be shuffled uniformly in each batch.
      dnatransform: Optional callable applied to the "sequences" variable.
    """
    def __init__(self, adata, batch_size=256, shuffle=True, dnatransform=None, shuffle_dims=None):
        self.adata = adata
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dnatransform = dnatransform
        self.shuffle_dims = shuffle_dims or []  # e.g. ['obs']
        self._generators = {}  # holds xbatcher generators for each state

    def setup(self, state="train"):
        """
        Set up the xbatcher generator for a given state.
        (It is assumed that the CrAnData object already has appropriate sample probabilities set.)
        """
        dim_dict = dict(self.adata.dims)
        # Override the "var" dimension with the desired batch size.
        dim_dict['var'] = self.batch_size
        dim_dict.pop('item', None)
        # Create the generator.
        self._generators[state] = xbatcher.BatchGenerator(
            ds=self.adata,
            input_dims=dim_dict,
            batch_dims={'var': self.batch_size},
        )

    def get_batch(self, state, index):
        """
        Retrieve a single batch (as produced by xbatcher) for the given state.
        Optionally apply the DNATransform to the "sequences" variable.
        Also applies per–dimension shuffling if requested.
        """
        if state not in self._generators:
            raise ValueError(f"Generator for state '{state}' not set. Call setup('{state}') first.")
        batch = self._generators[state][index]
        # Apply DNATransform to "sequences" if provided.
        if self.dnatransform is not None and 'sequences' in batch.data_vars:
            batch.data_vars['sequences'] = self.dnatransform(batch.data_vars['sequences'])
        # If shuffle is enabled and we have designated dimensions, shuffle them.
        if self.shuffle and self.shuffle_dims:
            for dim in self.shuffle_dims:
                # Get an example array to determine the axis.
                example = next((da for da in batch.data_vars.values() if dim in da.dims), None)
                if example is not None:
                    axis = example.get_axis_num(dim)
                    perm = np.random.permutation(example.shape[axis])
                    for var_name, da in batch.data_vars.items():
                        if dim in da.dims:
                            batch.data_vars[var_name] = da.isel({dim: perm})
        return batch

    # -------------------------------------------------------------------------
    # Dataloader properties (each returns a CrAnDataPipeLoader instance)
    # -------------------------------------------------------------------------
    @property
    def train_dataloader(self):
        return CrAnDataPipeLoader(self, state="train", collate_fn=self.batch_collate_fn,
                                  outer_batch_size=1, shuffle_obs=True, device=None)

    @property
    def val_dataloader(self):
        return CrAnDataPipeLoader(self, state="val", collate_fn=self.batch_collate_fn,
                                  outer_batch_size=1, shuffle_obs=True, device=None)

    @property
    def test_dataloader(self):
        return CrAnDataPipeLoader(self, state="test", collate_fn=self.batch_collate_fn,
                                  outer_batch_size=1, shuffle_obs=True, device=None)

    @property
    def predict_dataloader(self):
        return CrAnDataPipeLoader(self, state="predict", collate_fn=self.batch_collate_fn,
                                  outer_batch_size=1, shuffle_obs=True, device=None)

    def batch_collate_fn(self, batch):
        """
        A thin wrapper that uses the default collate function defined in CrAnDataPipeLoader.
        """
        loader = CrAnDataPipeLoader(self)
        loader.shuffle_obs = True
        loader.device = None
        return loader.default_batch_collate_fn(batch)

    def __repr__(self):
        return (f"CrAnDataModule(batch_size={self.batch_size}, shuffle_dims={self.shuffle_dims}, "
                f"dnatransform={self.dnatransform})")


class MetaCrAnDataModule:
    """
    Combines multiple CrAnData objects (each wrapped in a CrAnDataModule) into a single meta–module.
    For a given state the dataloader randomly samples one module (weighted by file probability)
    and then a batch from that module.
    
    Attributes:
      modules: List of CrAnDataModule instances.
      file_probs: Per–module sampling probabilities (here assumed uniform).
    """
    def __init__(self, adatas, batch_size=256, shuffle=True, dnatransform=None, shuffle_dims=None, epoch_size=100000):
        self.modules = [
            CrAnDataModule(adata, batch_size=batch_size, shuffle=shuffle, dnatransform=dnatransform, shuffle_dims=shuffle_dims)
            for adata in adatas
        ]
        # It is expected that setup(state) is called externally for each module.
        self.file_probs = np.ones(len(self.modules)) / len(self.modules)
        self.epoch_size = epoch_size

    @property
    def train_dataloader(self):
        return CrAnDataPipeLoader(self, state="train", collate_fn=self.batch_collate_fn,
                                  outer_batch_size=1, shuffle_obs=True, device=None, epoch_size=self.epoch_size)

    @property
    def val_dataloader(self):
        return CrAnDataPipeLoader(self, state="val", collate_fn=self.batch_collate_fn,
                                  outer_batch_size=1, shuffle_obs=True, device=None, epoch_size=self.epoch_size)

    @property
    def test_dataloader(self):
        return CrAnDataPipeLoader(self, state="test", collate_fn=self.batch_collate_fn,
                                  outer_batch_size=1, shuffle_obs=True, device=None, epoch_size=self.epoch_size)

    @property
    def predict_dataloader(self):
        return CrAnDataPipeLoader(self, state="predict", collate_fn=self.batch_collate_fn,
                                  outer_batch_size=1, shuffle_obs=True, device=None, epoch_size=self.epoch_size)

    def batch_collate_fn(self, batch):
        loader = CrAnDataPipeLoader(self)
        loader.shuffle_obs = True
        loader.device = None
        return loader.default_batch_collate_fn(batch)

    def __repr__(self):
        return (f"MetaCrAnDataModule(num_modules={len(self.modules)}, epoch_size={self.epoch_size})")
