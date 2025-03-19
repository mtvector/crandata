# CrAnData: Crested-style chromatin-adapted AnnData

Rebuilt lazy-loading AnnData-like datastructure on Xarray, backed by h5. Designed to generalize into multiple dimensions and allow for dataloading of other AnnData fields, like layers, obsm, etc.

Proof of concept works as a dataloader, but doesn't play nicely with crested's fit() currently. 

See [this notebook](tests/test_crandataloader.ipynb) for examples on how to create and use the class.

There may or may not be further development of this class...
