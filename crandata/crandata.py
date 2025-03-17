import xarray as xr
import pandas as pd
import numpy as np
import os
import json

try:
    import sparse  # for sparse multidimensional arrays
except ImportError:
    print("no sparse")
    sparse = None

class CrAnData(xr.Dataset):
    # No __slots__; all attributes are stored in __dict__
    
    def __init__(self, 
                 data_vars=None,
                 coords=None,
                 always_convert_df=[],  # list of top-level keys to be grouped into a DataFrame on access
                 **kwargs):
        """
        Create a CrAnData object as a subclass of xarray.Dataset.
        """
        if data_vars is None:
            data_vars = {}
        # Make a copy and merge any additional kwargs.
        data_vars = dict(data_vars)
        data_vars.update(kwargs)
        # Ensure every variable is an xr.DataArray.
        for key, var in data_vars.items():
            if not isinstance(var, xr.DataArray):
                data_vars[key] = xr.DataArray(var)
        if coords is None:
            coords = {}
        
        super().__init__(data_vars=data_vars, coords=coords)
        

        self.always_convert_df = always_convert_df
        self.attrs["always_convert_df"] = json.dumps(self.always_convert_df)
        
        # For keys that contain '/', create shortcut attributes.
        for key in self.data_vars:
            if "/" in key:
                safe_name = key.replace("/", "_")
                setattr(self, safe_name, self.data_vars[key])

        if "var" in self.always_convert_df:
            grouped_var = self.get_dataframe("var")
            if grouped_var is not None:
                self.var = grouped_var  # shadow built-in var
    
    @property
    def array_names(self):
        return list(self.data_vars.keys())
    
    def get_dataframe(self, top):
        cols = {}
        expected = None
        for key in list(self.data_vars.keys()):
            if key.startswith(top + "/"):
                da = super().__getitem__(key)
                if expected is None:
                    expected = da.shape[0]
                if da.shape[0] != expected:
                    continue  # skip keys that do not match expected length
                col_name = key.split("/", 1)[1]
                cols[col_name] = da.values
        if cols:
            return pd.DataFrame(cols, index=np.arange(expected))
        else:
            return None
    
    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self.always_convert_df:
                df = self.get_dataframe(key)
                if df is None:
                    raise KeyError(f"No grouped data found for key '{key}'")
                return df
            if "/" in key:
                top, sub = key.split("/", 1)
                if top in self.always_convert_df:
                    df = self.get_dataframe(top)
                    if df is None:
                        raise KeyError(f"No grouped data found for key '{top}'")
                    return df[sub]
        return super().__getitem__(key)
    
    def __getattr__(self, attr):
        always_convert = object.__getattribute__(self, "always_convert_df")
        if attr in always_convert:
            df = self.get_dataframe(attr)
            if df is not None:
                return df
        dv = object.__getattribute__(self, "data_vars")
        for key in dv:
            safe = key.replace("/", "_")
            if safe == attr:
                return dv[key]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {attr!r}")
    
    def __repr__(self):
        rep = f"CrAnData object\nArray names: {self.array_names}\n"
        rep += f"Coordinates: {list(self.coords.keys())}\n"
        return rep
        
    def _repr_html_(self):
        return self.__repr__()
    
    # === Sparse encoding/decoding methods ===
    @staticmethod
    def _encode_sparse(var):
        """
        Encode a sparse.COO array (from var.data) into a JSON-serializable dictionary.
        """
        sp = var.data  # assume sparse.COO
        return {
            "sparse": True,
            "data": sp.data.tolist(),
            "coords": sp.coords.tolist(),
            "shape": list(sp.shape),
            "dtype": str(sp.dtype)
        }
    
    @staticmethod
    def _decode_sparse(encoded):
        """
        Rebuild a sparse.COO array from its encoded dictionary.
        """
        data = np.array(encoded["data"])
        coords = np.array(encoded["coords"])
        shape = tuple(encoded["shape"])
        return sparse.COO(coords, data, shape=shape)
    
    def sparse_serialized(self):
        """
        Return a new CrAnData object with all data variables whose underlying data is a sparse.COO
        replaced by encoded versions stored as new variables named "encoded_<key>".
        The original keys are dropped.
        """
        if sparse is None:
            return self
        encoded_vars = {}
        keys_to_drop = []
        for key, var in list(self.data_vars.items()):
            try:
                data_attr = var.data
            except Exception:
                continue
            if hasattr(data_attr, "todense") and isinstance(data_attr, sparse.COO):
                encoded_vars[key] = json.dumps(self._encode_sparse(var))
                keys_to_drop.append(key)
        # Create a new dataset without the sparse variables.
        new_ds = self.drop_vars(keys_to_drop)
        # Add the encoded variables.
        for key, encoded_str in encoded_vars.items():
            new_ds = new_ds.assign({ "encoded_" + key: xr.DataArray(encoded_str, dims=()) })
        if encoded_vars:
            new_ds.attrs["sparse_encoded_keys"] = json.dumps(list(encoded_vars.keys()))
        # Preserve always_convert_df in attrs.
        new_ds.attrs["always_convert_df"] = json.dumps(self.always_convert_df)
        new_ds.always_convert_df = self.always_convert_df
        return new_ds
    
    @classmethod
    def _decode_sparse_from_vars(cls, obj):
        # Build a new dictionary for data_vars.
        new_vars = dict(obj.data_vars)
        for key in list(new_vars):
            if key.startswith("encoded_"):
                orig_key = key[len("encoded_"):]
                encoded_str = new_vars[key].values.item()
                enc = json.loads(encoded_str)
                decoded = cls._decode_sparse(enc)
                new_vars[orig_key] = xr.DataArray(decoded)
                del new_vars[key]
        always_convert = json.loads(obj.attrs.get("always_convert_df", "[]"))
        return cls(data_vars=new_vars, coords=obj.coords, always_convert_df=always_convert)
    
    @classmethod
    def open_dataset(cls, path, **kwargs):
        ds = xr.open_dataset(path, **kwargs)
        always_convert_df = json.loads(ds.attrs.get("always_convert_df", "[]"))
        obj = cls(data_vars=ds.data_vars, coords=ds.coords, always_convert_df=always_convert_df)
        obj = cls._decode_sparse_from_vars(obj)
        return obj

    @classmethod
    def open_zarr(cls, store, **kwargs):
        ds = xr.open_zarr(store, **kwargs)
        always_convert_df = json.loads(ds.attrs.get("always_convert_df", "[]"))
        obj = cls(data_vars=ds.data_vars, coords=ds.coords, always_convert_df=always_convert_df)
        obj = cls._decode_sparse_from_vars(obj)
        return obj
