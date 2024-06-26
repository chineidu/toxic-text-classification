from typing import Any, Callable

import lightning as l
import numpy as np
import numpy.typing as npt
from torch import Tensor
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    Sampler,
    default_collate,
    random_split,
)
from transformers import BatchEncoding
from typeguard import typechecked

from toxic_classifier.datamodule.dataset import TextDataset
from toxic_classifier.ml.transformations import HuggingFaceTokenizationTransformation


class BaseDataModule(l.LightningDataModule):
    """
    This is the base class for all DataModules.
    """

    def __init__(
        self,
        batch_size: int,
        sampler: Sampler | None = None,
        batch_sampler: BatchSampler | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[Any], Any] | None = None,
        shuffle: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers

    def init_dataloader(self, dataset: Dataset, is_test: bool = False) -> DataLoader:
        """This is used to initialize the dataloader."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=not self.shuffle and not is_test,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )


class TextDataModule(BaseDataModule):
    """This is used to load the data."""

    @typechecked
    def __init__(
        self,
        text_features: list[str],
        labels: npt.NDArray[np.float_],
        batch_size: int,
        transformation: HuggingFaceTokenizationTransformation,
        sampler: Sampler | None = None,
        batch_sampler: BatchSampler | None = None,
        num_workers: int = 0,
        shuffle: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        def tokenization_collate_fn(batch: list[tuple[str, int]]) -> tuple[BatchEncoding, Tensor]:
            """This is used to collate the tokenized texts."""
            texts, labels = default_collate(batch)
            encodings = transformation(texts)
            return encodings, labels

        super().__init__(
            batch_size=batch_size,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=tokenization_collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )
        self.features = text_features
        self.labels = labels

    @typechecked
    def prepare_data(self) -> None:
        """This is used to download the data."""
        pass

    @typechecked
    def setup(self, stage: str) -> None:
        text_data: Dataset = TextDataset(text_features=self.features, labels=self.labels)
        size_a: int = int(len(text_data) * 0.85)
        size_b: int = len(text_data) - size_a
        train_data, test_data = random_split(dataset=text_data, lengths=[size_a, size_b])

        size_a: int = int(len(train_data) * 0.85)  # type: ignore
        size_b: int = len(train_data) - size_a  # type: ignore
        train_data, val_data = random_split(dataset=train_data, lengths=[size_a, size_b])
        if stage == "fit" or stage is None:
            self.train_data = train_data
            self.val_data = val_data
        if stage == "test":
            self.test_data = test_data

    @typechecked
    def train_dataloader(self) -> DataLoader:
        return self.init_dataloader(dataset=self.train_data, is_test=False)

    @typechecked
    def val_dataloader(self) -> DataLoader:
        return self.init_dataloader(dataset=self.val_data, is_test=True)

    @typechecked
    def test_dataloader(self) -> DataLoader:
        return self.init_dataloader(dataset=self.test_data, is_test=True)
