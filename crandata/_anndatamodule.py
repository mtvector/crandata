"""_anndatamodule.py – Module for wrapping Yanndata-based AnnDataset and AnnDataLoader.

This module packages the dataset, dataloader, and sampler functionality so that you can
load data from your Yanndata files.
"""

from __future__ import annotations
from os import PathLike
import numpy as np
from ._genome import Genome, _resolve_genome #Just copied from crested
from anndata import AnnData
from .yanndata import CrAnData
from ._dataloader import AnnDataLoader
from ._dataset import AnnDataset, MetaAnnDataset

def set_stage_sample_probs(adata: CrAnData, stage: str):
    required_cols = ["split"]
    for c in required_cols:
        if c not in adata.var:
            raise KeyError(f"Missing column {c} in adata.var")
    sample_probs = np.zeros(adata.n_vars, dtype=float)
    if stage == "train":
        mask = (adata.var["split"] == "train")
        if "train_probs" not in adata.var:
            adata.var["train_probs"] = 1.0
        adata.var["train_probs"] = adata.var["train_probs"] / adata.var["train_probs"].sum()
        sample_probs[mask] = adata.var["train_probs"][mask].values
        adata.var["sample_probs"] = sample_probs / sample_probs.sum()
        mask = (adata.var["split"] == "val")
        adata.var["val_probs"] = mask.astype(float)
        adata.var["val_probs"] = adata.var["val_probs"] / adata.var["val_probs"].sum()
    elif stage == "test":
        mask = (adata.var["split"] == "test")
        adata.var["test_probs"] = mask.astype(float)
        adata.var["test_probs"] = adata.var["test_probs"] / adata.var["test_probs"].sum()
    elif stage == "predict":
        adata.var["predict_probs"] = 1.0
        adata.var["predict_probs"] = adata.var["predict_probs"] / adata.var["predict_probs"].sum()
    else:
        print("Invalid stage, sample probabilities unchanged")

class AnnDataModule:
    """
    DataModule that wraps an AnnData (CrAnData) object using AnnDataset with a unified data_sources interface.
    
    Parameters:
      adata : CrAnData
          CrAnData object containing the data.
      genome : PathLike | Genome | None
          Genome instance or FASTA path.
      chromsizes_file : PathLike | None
          Path to chromsizes file.
      in_memory : bool
          If True, load sequences into memory.
      always_reverse_complement : bool
          If True, always add reverse complement sequences.
      random_reverse_complement : bool
          If True, randomly reverse complement during training.
      max_stochastic_shift : int
          Maximum random shift.
      deterministic_shift : bool
          Use legacy shifting if True.
      shuffle : bool
          Whether to shuffle training data.
      shuffle_obs : bool
          Whether to shuffle the obs dimension of each batch
      batch_size : int
          Samples per batch.
      data_sources : dict[str, str]
          Mapping of keys to data sources.
    """
    def __init__(
        self,
        adata: CrAnData,
        genome: PathLike | Genome | None = None,
        chromsizes_file: PathLike | None = None,
        in_memory: bool = True,
        always_reverse_complement: bool = True,
        random_reverse_complement: bool = False,
        max_stochastic_shift: int = 0,
        deterministic_shift: bool = False,
        shuffle: bool = True,
        batch_size: int = 256,
        data_sources: dict[str, str] = {'y': 'X'},
    ):
        self.adata = adata
        self.genome = _resolve_genome(genome, chromsizes_file)
        self.in_memory = in_memory
        self.always_reverse_complement = always_reverse_complement
        self.random_reverse_complement = random_reverse_complement
        self.max_stochastic_shift = max_stochastic_shift
        self.deterministic_shift = deterministic_shift
        self.shuffle = shuffle
        self.shuffle_obs = shuffle_obs
        self.batch_size = batch_size
        self.data_sources = data_sources

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    @staticmethod
    def _split_anndata(adata: CrAnData, split: str) -> CrAnData:
        if split:
            if "split" not in adata.var.columns:
                raise KeyError("No split column found in adata.var. Run the appropriate pre-processing.")
        return adata

    def setup(self, stage: str) -> None:
        args = {
            "adata": self.adata,
            "genome": self.genome,
            "data_sources": self.data_sources,
            "in_memory": self.in_memory,
            "always_reverse_complement": self.always_reverse_complement,
            "random_reverse_complement": self.random_reverse_complement,
            "max_stochastic_shift": self.max_stochastic_shift,
            "deterministic_shift": self.deterministic_shift,
            "split": None,
        }
        if stage == "fit":
            train_args = args.copy()
            train_args["split"] = "train"
            set_stage_sample_probs(self.adata, "train")
            self.train_dataset = AnnDataset(**train_args)
            val_args = args.copy()
            val_args["split"] = "val"
            val_args["always_reverse_complement"] = False
            val_args["random_reverse_complement"] = False
            val_args["max_stochastic_shift"] = 0
            self.val_dataset = AnnDataset(**val_args)
        elif stage == "test":
            test_args = args.copy()
            test_args["split"] = "test"
            test_args["in_memory"] = False
            test_args["always_reverse_complement"] = False
            test_args["random_reverse_complement"] = False
            test_args["max_stochastic_shift"] = 0
            set_stage_sample_probs(self.adata, "test")
            self.test_dataset = AnnDataset(**test_args)
        elif stage == "predict":
            predict_args = args.copy()
            predict_args["split"] = None
            predict_args["in_memory"] = False
            predict_args["always_reverse_complement"] = False
            predict_args["random_reverse_complement"] = False
            predict_args["max_stochastic_shift"] = 0
            set_stage_sample_probs(self.adata, "predict")
            self.predict_dataset = AnnDataset(**predict_args)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    @property
    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Train dataset not set. Run setup('fit') first.")
        return AnnDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            shuffle_obs=self.shuffle_obs,
            drop_remainder=False,
            stage='train'
        )

    @property
    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Val dataset not set. Run setup('fit') first.")
        return AnnDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
            stage='val'
        )

    @property
    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset not set. Run setup('test') first.")
        return AnnDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
            stage='test'
        )

    @property
    def predict_dataloader(self):
        if self.predict_dataset is None:
            raise ValueError("Predict dataset not set. Run setup('predict') first.")
        return AnnDataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
            stage='predict'
        )

    def __repr__(self):
        return (f"AnnDataModule(adata_shape={self.adata.shape}, genome={self.genome}, "
                f"in_memory={self.in_memory}, always_reverse_complement={self.always_reverse_complement}, "
                f"random_reverse_complement={self.random_reverse_complement}, max_stochastic_shift={self.max_stochastic_shift}, "
                f"shuffle={self.shuffle}, batch_size={self.batch_size})")


class MetaAnnDataModule:
    """
    DataModule for combining multiple AnnData objects (e.g. one per species)
    into a single MetaAnnDataset for global weighted sampling.

    Each AnnData (ideally a CrAnData) is first wrapped in an AnnDataset, and then the 
    resulting datasets are merged into a MetaAnnDataset.

    Parameters
    ----------
    adatas : list[CrAnData]
        Each species/dataset stored in its own CrAnData.
    genomes : list[Genome]
        Matching list of genome references.
    data_sources : dict[str, str], default {'y': 'X'}
        Mapping of keys to data sources.
    in_memory : bool, default True
        Whether to load sequences into memory.
    random_reverse_complement : bool, default True
        Whether to randomly reverse complement each region.
    max_stochastic_shift : int, default 0
        Maximum shift (±bp) for augmentation.
    deterministic_shift : bool, default False
        If True, apply legacy fixed-stride shifting.
    shuffle : bool, default True
        Whether to shuffle the dataset.
    batch_size : int, default 256
        Number of samples per batch.
    epoch_size : int, default 100_000
        Number of samples per epoch for custom sampling.
    """
    def __init__(
        self,
        adatas: list[CrAnData],
        genomes: list[Genome],
        data_sources: dict[str, str] = {'y': 'X'},
        in_memory: bool = True,
        random_reverse_complement: bool = True,
        max_stochastic_shift: int = 0,
        deterministic_shift: bool = False,
        shuffle: bool = True,
        shuffle_obs: bool = False,
        batch_size: int = 256,
        epoch_size: int = 100_000,
        obs_alignment: str = 'union',
    ):
        if len(adatas) != len(genomes):
            raise ValueError("Must provide as many adatas as genomes.")
        
        self.adatas = adatas
        self.genomes = genomes
        self.in_memory = in_memory
        self.random_reverse_complement = random_reverse_complement
        self.max_stochastic_shift = max_stochastic_shift
        self.deterministic_shift = deterministic_shift
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.data_sources = data_sources
        self.epoch_size = epoch_size
        self.shuffle_obs = shuffle_obs

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        
        # Compute observation names alignment across adatas:
        if obs_alignment == 'union':
            meta_obs_names = np.array(set().union(*[set(adata.obs_names) for adata in self.adatas]))
        elif obs_alignment == 'intersect':
            meta_obs_names = np.array(set.intersection(*[set(adata.obs_names) for adata in self.adatas]))
        else:
            raise ValueError("obs_alignment must be 'union' or 'intersect'")
        self.meta_obs_names = meta_obs_names
        for adata in self.adatas:
            adata.meta_obs_names = self.meta_obs_names

    def setup(self, stage: str) -> None:
        def dataset_args(split):
            return {
                "in_memory": self.in_memory,
                "data_sources": self.data_sources,
                "always_reverse_complement": False,  # Disable augmentation by default in meta mode
                "random_reverse_complement": self.random_reverse_complement,
                "max_stochastic_shift": self.max_stochastic_shift,
                "deterministic_shift": self.deterministic_shift,
                "split": split,
            }
        if stage == "fit":
            train_datasets = []
            val_datasets = []
            for adata, genome in zip(self.adatas, self.genomes):
                args = dataset_args("train")
                set_stage_sample_probs(adata, "train")
                ds_train = AnnDataset(adata=adata, genome=genome, **args)
                train_datasets.append(ds_train)

                val_args = dataset_args("val")
                val_args["always_reverse_complement"] = False
                val_args["random_reverse_complement"] = False
                val_args["max_stochastic_shift"] = 0
                ds_val = AnnDataset(adata=adata, genome=genome, **val_args)
                val_datasets.append(ds_val)
            self.train_dataset = MetaAnnDataset(train_datasets)
            self.val_dataset = MetaAnnDataset(val_datasets)

        elif stage == "test":
            test_datasets = []
            for adata, genome in zip(self.adatas, self.genomes):
                args = dataset_args("test")
                set_stage_sample_probs(adata, "test")
                args["in_memory"] = False
                args["always_reverse_complement"] = False
                args["random_reverse_complement"] = False
                args["max_stochastic_shift"] = 0
                ds_test = AnnDataset(adata=adata, genome=genome, **args)
                test_datasets.append(ds_test)
            self.test_dataset = MetaAnnDataset(test_datasets)

        elif stage == "predict":
            predict_datasets = []
            for adata, genome in zip(self.adatas, self.genomes):
                args = dataset_args(None)
                set_stage_sample_probs(adata, "predict")
                args["in_memory"] = False
                args["always_reverse_complement"] = False
                args["random_reverse_complement"] = False
                args["max_stochastic_shift"] = 0
                ds_pred = AnnDataset(adata=adata, genome=genome, **args)
                predict_datasets.append(ds_pred)
            self.predict_dataset = MetaAnnDataset(predict_datasets)

        else:
            raise ValueError(f"Invalid stage: {stage}")

    @property
    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("train_dataset is not set. Run setup('fit') first.")
        return AnnDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            shuffle_obs=self.shuffle_obs,
            drop_remainder=False,
            epoch_size=self.epoch_size,
            stage='train'
        )

    @property
    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("val_dataset is not set. Run setup('fit') first.")
        return AnnDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
            epoch_size=self.epoch_size,
            stage='val'
        )

    @property
    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("test_dataset is not set. Run setup('test') first.")
        return AnnDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
            epoch_size=self.epoch_size,
            stage='test'
        )

    @property
    def predict_dataloader(self):
        if self.predict_dataset is None:
            raise ValueError("predict_dataset is not set. Run setup('predict') first.")
        return AnnDataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
            epoch_size=self.epoch_size,
            stage='predict'
        )

    def __repr__(self):
        return (
            f"MetaAnnDataModule(num_species={len(self.adatas)}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, max_stochastic_shift={self.max_stochastic_shift}, "
            f"random_reverse_complement={self.random_reverse_complement}, "
            f"always_reverse_complement={self.always_reverse_complement}, in_memory={self.in_memory}, "
            f"deterministic_shift={self.deterministic_shift}, epoch_size={self.epoch_size})"
        )
