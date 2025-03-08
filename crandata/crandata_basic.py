'''yanndata.py'''

from anndata import AnnData
import xarray as xr
import pandas as pd
import numpy as np

try:
    import sparse  # optional: for sparse multidimensional arrays
except ImportError:
    print("no sparse")
    sparse = None

def index_dim(xarr, dim, key):
    # If key is an int, slice, or list/tuple of ints, use isel (positional indexing)
    if isinstance(key, (int, slice, np.integer)) or \
       (isinstance(key, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in key)):
        return xarr.isel({dim: key})
    else:
        # Otherwise, use coordinate-based selection.
        return xarr.sel({dim: key})

class _XWrapper:
    """
    Wraps an xarray DataArray (with proper coordinates) to allow unified indexing:
    if the key for a given dimension is an integer or slice, use .isel,
    otherwise use .sel.
    Missing dimensions are automatically filled with full slices.
    """
    def __init__(self, xarray_obj):
        self._x = xarray_obj

    def __getitem__(self, key):
        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)
        # Append full slices if fewer keys than dims.
        if len(key) < self._x.ndim:
            key = key + (slice(None),) * (self._x.ndim - len(key))
        result = self._x
        for dim, k in zip(self._x.dims, key):
            result = index_dim(result, dim, k)
        return result

    def __setitem__(self, key, value):
        # For simplicity, use the same indexing behavior for setting.
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) < self._x.ndim:
            key = key + (slice(None),) * (self._x.ndim - len(key))
        # We have to decide per dimension whether to use .isel or .sel.
        # Here we convert key to a dict for isel if all keys are ints/slices.
        if all(isinstance(k, (int, slice, np.integer)) for k in key):
            self._x = self._x.copy(data=self._x.isel(dict(zip(self._x.dims, key))).data)
        else:
            # Otherwise, we index iteratively.
            result = self._x
            for dim, k in zip(self._x.dims, key):
                result = index_dim(result, dim, k)
            # Replace the data in the result.
            self._x = result
        # Note: More sophisticated updating may be needed for in-place modification.

    def __array__(self):
        return np.asarray(self._x)

    def __repr__(self):
        return repr(self._x)


def _to_dataarray(X, dim_names):
    """Convert an array (or xarray DataArray) to an xarray DataArray with given dims."""
    if not isinstance(X, xr.DataArray):
        X = np.asarray(X)
        X = xr.DataArray(X, dims=dim_names)
    return X


def _convert_to_varp(varp, n_var):
    """Convert varp to a DataArray with dims ["var_0", "var_1", ...]."""
    if not isinstance(varp, xr.DataArray):
        varp = np.asarray(varp)
        extra = varp.ndim - 2
        dims = ["var_0", "var_1"] + [f"dim_{i}" for i in range(extra)]
        varp = xr.DataArray(varp, dims=dims)
    if varp.sizes["var_0"] != n_var or varp.sizes["var_1"] != n_var:
        raise ValueError("The first two dimensions of varp must match the number of variables.")
    return varp


def _convert_to_obsp(obsp, n_obs):
    """Convert obsp to a DataArray with dims ["obs_0", "obs_1", ...]."""
    if not isinstance(obsp, xr.DataArray):
        obsp = np.asarray(obsp)
        extra = obsp.ndim - 2
        dims = ["obs_0", "obs_1"] + [f"dim_{i}" for i in range(extra)]
        obsp = xr.DataArray(obsp, dims=dims)
    if obsp.sizes["obs_0"] != n_obs or obsp.sizes["obs_1"] != n_obs:
        raise ValueError("The first two dimensions of obsp must match the number of observations.")
    return obsp


class CrAnData(AnnData):
    """
    A minimal drop-in replacement for anndata.AnnData using xarray.
    
    - The primary data (X) is stored as an xarray DataArray with dimensions
      "obs", "var", and any extra dimensions. The "obs" and "var" dimensions
      have coordinates taken from the obs and var DataFrame indices.
    - varp and obsp (if provided) are stored as multidimensional DataArrays with
      their first two dimensions labeled "var_0"/"var_1" and "obs_0"/"obs_1", and
      coordinates from var and obs respectively.
    - obsm and varm are dictionaries of 2D arrays that are stored as DataArrays
      with dims ["obs", "col"] or ["var", "col"] and an appropriate coordinate for
      the first dimension.
      
    Indexing into X, varp, obsp, obsm, and varm supports both integer (positional)
    indexing and coordinate-based indexing.
    """
    def __init__(self, X, obs=None, var=None, uns=None,
                 obsm=None, varm=None, layers=None,
                 varp=None, obsp=None):
        # Bypass superclass initialization.
        try:
            super().__init__(None)
        except Exception:
            print("I'm not a real AnnData... Bypassing AnnData.__init__")
        # Convert X to an xarray DataArray.
        arr = np.asarray(X)
        extra = arr.ndim - 2
        dims = ["obs", "var"] + [f"dim_{i}" for i in range(extra)]
        self._X = _to_dataarray(arr, dims)

        # Set up obs and var DataFrames.
        n_obs, n_var = self._X.sizes["obs"], self._X.sizes["var"]
        if obs is None:
            self._obs = pd.DataFrame(index=[str(i) for i in range(n_obs)])
        else:
            self._obs = pd.DataFrame(obs)
            if len(self._obs) != n_obs:
                raise ValueError("Length of obs does not match number of observations in X.")
        if var is None:
            self._var = pd.DataFrame(index=[str(i) for i in range(n_var)])
        else:
            self._var = pd.DataFrame(var)
            if len(self._var) != n_var:
                raise ValueError("Length of var does not match number of variables in X.")

        # Assign coordinates for X.
        self._X = self._X.assign_coords(obs=self._obs.index, var=self._var.index)

        self.uns = uns if uns is not None else {}
        self.layers = layers if layers is not None else {}

        # Set obsm and varm (stored as dictionaries of 2D DataArrays).
        self.obsm = obsm if obsm is not None else {}
        self.varm = varm if varm is not None else {}

        # Set varp and obsp, assigning coordinates.
        if varp is not None:
            vp = _convert_to_varp(varp, n_var)
            self._varp = vp.assign_coords(var_0=self._var.index, var_1=self._var.index)
        else:
            self._varp = None

        if obsp is not None:
            op = _convert_to_obsp(obsp, n_obs)
            self._obsp = op.assign_coords(obs_0=self._obs.index, obs_1=self._obs.index)
        else:
            self._obsp = None

    @property
    def X(self):
        return _XWrapper(self._X)

    @X.setter
    def X(self, value):
        self._X = _to_dataarray(np.asarray(value), self._X.dims)
        # Reassign coordinates for obs and var.
        self._X = self._X.assign_coords(obs=self._obs.index, var=self._var.index)

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, value):
        df = pd.DataFrame(value)
        if len(df) != self._X.sizes["obs"]:
            raise ValueError("Length of new obs does not match number of observations in X.")
        self._obs = df
        # Update coordinates in X and obsm.
        self._X = self._X.assign_coords(obs=self._obs.index)
        if self._obsp is not None:
            self._obsp = self._obsp.assign_coords(obs_0=self._obs.index, obs_1=self._obs.index)
        # Also update obsm entries.
        for k, arr in self.obsm.items():
            self.obsm[k] = xr.DataArray(arr.data, dims=["obs", "col"],
                                          coords={"obs": self._obs.index, "col": np.arange(arr.shape[1])})

    @property
    def var(self):
        return self._var

    @var.setter
    def var(self, value):
        df = pd.DataFrame(value)
        if len(df) != self._X.sizes["var"]:
            raise ValueError("Length of new var does not match number of variables in X.")
        self._var = df
        # Update coordinates in X and varm.
        self._X = self._X.assign_coords(var=self._var.index)
        if self._varp is not None:
            self._varp = self._varp.assign_coords(var_0=self._var.index, var_1=self._var.index)
        # Also update varm entries.
        for k, arr in self.varm.items():
            self.varm[k] = xr.DataArray(arr.data, dims=["var", "col"],
                                          coords={"var": self._var.index, "col": np.arange(arr.shape[1])})

    @property
    def obsm(self):
        return self._obsm

    @obsm.setter
    def obsm(self, value):
        validated = {}
        for k, v in (value or {}).items():
            arr = np.asarray(v)
            if arr.ndim != 2:
                raise ValueError(f"obsm entry '{k}' must be 2D.")
            if arr.shape[0] != self._obs.shape[0]:
                raise ValueError(f"obsm entry '{k}' first dimension must match number of observations.")
            xarr = xr.DataArray(arr, dims=["obs", "col"],
                                coords={"obs": self._obs.index, "col": np.arange(arr.shape[1])})
            validated[k] = xarr
        self._obsm = validated

    @property
    def varm(self):
        return self._varm

    @varm.setter
    def varm(self, value):
        validated = {}
        for k, v in (value or {}).items():
            arr = np.asarray(v)
            if arr.ndim != 2:
                raise ValueError(f"varm entry '{k}' must be 2D.")
            if arr.shape[0] != self._var.shape[0]:
                raise ValueError(f"varm entry '{k}' first dimension must match number of variables.")
            xarr = xr.DataArray(arr, dims=["var", "col"],
                                coords={"var": self._var.index, "col": np.arange(arr.shape[1])})
            validated[k] = xarr
        self._varm = validated

    @property
    def varp(self):
        return self._varp

    @varp.setter
    def varp(self, value):
        n_var = self._X.sizes["var"]
        vp = _convert_to_varp(value, n_var)
        self._varp = vp.assign_coords(var_0=self._var.index, var_1=self._var.index)

    @property
    def obsp(self):
        return self._obsp

    @obsp.setter
    def obsp(self, value):
        n_obs = self._X.sizes["obs"]
        op = _convert_to_obsp(value, n_obs)
        self._obsp = op.assign_coords(obs_0=self._obs.index, obs_1=self._obs.index)

    @property
    def shape(self):
        return self._X.shape

    def copy(self):
        return CrAnData(
            self._X.copy(),
            obs=self._obs.copy(),
            var=self._var.copy(),
            uns=self.uns.copy(),
            obsm={k: v.copy() for k, v in self.obsm.items()},
            varm={k: v.copy() for k, v in self.varm.items()},
            layers=self.layers.copy(),
            varp=self._varp.copy() if self._varp is not None else None,
            obsp=self._obsp.copy() if self._obsp is not None else None,
        )

    def __getitem__(self, key):
        """
        Slices the AnnData object along the obs and var dimensions.
        Supports both integer and label indexing.
        """
        if not isinstance(key, tuple):
            key = (key, slice(None))
        if len(key) != 2:
            raise IndexError("Only two indices (obs, var) are supported for slicing.")
        obs_key, var_key = key

        new_X = self._X
        new_X = index_dim(new_X, "obs", obs_key)
        new_X = index_dim(new_X, "var", var_key)
        new_obs = self._subset_df(self._obs, obs_key)
        new_var = self._subset_df(self._var, var_key)
        new_varp = self._varp
        if new_varp is not None:
            new_varp = index_dim(new_varp, "var_0", var_key)
            new_varp = index_dim(new_varp, "var_1", var_key)
        new_obsp = self._obsp
        if new_obsp is not None:
            new_obsp = index_dim(new_obsp, "obs_0", obs_key)
            new_obsp = index_dim(new_obsp, "obs_1", obs_key)
        new_obsm = {k: index_dim(v, "obs", obs_key) for k, v in self.obsm.items()}
        new_varm = {k: index_dim(v, "var", var_key) for k, v in self.varm.items()}

        return CrAnData(
            new_X,
            obs=new_obs,
            var=new_var,
            uns=self.uns.copy(),
            obsm=new_obsm,
            varm=new_varm,
            layers=self.layers.copy(),
            varp=new_varp,
            obsp=new_obsp,
        )

    def _subset_df(self, df, key):
        if isinstance(key, (int, np.integer)):
            return df.iloc[[key]]
        elif isinstance(key, slice):
            return df.iloc[key]
        else:
            # If key is label(s), use .loc
            return df.loc[key]

    def __repr__(self):
        n_obs = self._X.sizes["obs"]
        n_var = self._X.sizes["var"]
        return f"CrAnData object with {n_obs} observations and {n_var} variables"