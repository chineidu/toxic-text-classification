from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl
from typeguard import typechecked

from toxic_classifier import config


class DataLoader(ABC):
    target: str = config.constants.target
    seed: int = config.constants.seed
    test_size: float = config.constants.test_size

    @abstractmethod
    def __init__(self, path: str | Path, separator: str) -> None:
        """DataLoader constructor."""

        self.path = path
        self.separator = separator

    def load_data(self) -> pl.DataFrame:
        """This is used to load the data."""
        raise NotImplementedError()

    @abstractmethod
    def add_target(self) -> pl.DataFrame:
        """This is used to add the target label to the data."""
        raise NotImplementedError()

    @abstractmethod
    def dataset_name(self) -> pl.DataFrame:
        """This is used to add the dataset name to the data."""
        raise NotImplementedError()

    @abstractmethod
    def prepare_data(
        self,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """This is used to prepare the data."""
        raise NotImplementedError()


class GHCDataLoader(DataLoader):
    @typechecked
    def __init__(self, path: str | Path, separator: str = "\t") -> None:
        """Gabe Hate Corpus DataLoader."""
        super().__init__(path, separator)

    @typechecked
    def load_data(self) -> pl.DataFrame:
        return pl.read_csv(self.path, separator=self.separator)

    @typechecked
    def dataset_name(self) -> pl.DataFrame:
        data: pl.DataFrame = self.load_data()
        data = data.with_columns(dataset=pl.lit("gabe_hate_corpus"))
        return data

    @typechecked
    def add_target(self) -> pl.DataFrame:
        data: pl.DataFrame = self.dataset_name()
        # Add target label
        data = data.with_columns(pl.any_horizontal("hd", "cv", "vo").alias(self.target))
        data = data.select(["text", "dataset", self.target])
        return data

    @typechecked
    def prepare_data(self) -> pl.DataFrame:
        data: pl.DataFrame = self.add_target()
        return data


class ToxicCommentsDataLoader(DataLoader):
    @typechecked
    def __init__(
        self,
        path: str | Path,
        separator: str = ",",
        other_path: None | str | Path = None,
        labels_path: None | str | Path = None,
    ) -> None:
        """Toxic Comments DataLoader."""
        super().__init__(path, separator)
        self.other_path = other_path
        self.labels_path = labels_path

    @typechecked
    def load_data(self) -> pl.DataFrame:
        return pl.read_csv(self.path, separator=self.separator)

    @typechecked
    def _load_test_labels(self) -> pl.DataFrame:
        """This is used to load the test labels."""
        return pl.read_csv(self.labels_path, separator=self.separator)

    @typechecked
    def _load_test_data(self) -> pl.DataFrame:
        """This is used to load the test data."""
        return pl.read_csv(self.other_path, separator=self.separator)

    @typechecked
    def _join_data(self) -> pl.DataFrame:
        """This is used to join the test data and labels."""
        test_data: pl.DataFrame = self._load_test_data()
        test_labels: pl.DataFrame = self._add_target()
        # Merge the data
        test_data = test_data.join(test_labels, on="id")
        return test_data

    @typechecked
    def _dataset_name(self) -> pl.DataFrame:
        """This is used to add the dataset name to the labels data."""
        data: pl.DataFrame = self._load_test_labels()
        data = data.with_columns(dataset=pl.lit("toxic_comments"))
        return data

    # @typechecked
    def dataset_name(self) -> pl.DataFrame:
        pass
        data: pl.DataFrame = self.load_data()
        data = data.with_columns(dataset=pl.lit("toxic_comments"))
        return data

    @typechecked
    def _add_target(self) -> pl.DataFrame:
        """Create the target column for the test data."""
        data: pl.DataFrame = self._dataset_name()
        # Add target label
        data = data.with_columns(
            pl.mean_horizontal(
                "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
            )
            .round(2)
            .alias("contains_scoring_data")
        )
        # Drop the labels with `-1`
        data = data.filter(pl.col("contains_scoring_data").ne(-1.0))
        data = data.with_columns(
            pl.when(pl.col("contains_scoring_data").eq(0.0))
            .then(pl.lit(False))
            .otherwise(pl.lit(True))
            .alias(self.target)
        )
        data = data.select(["id", "dataset", self.target])
        return data

    @typechecked
    def add_target(self) -> pl.DataFrame:
        data: pl.DataFrame = self.dataset_name()
        # Add target label
        data = data.with_columns(
            pl.mean_horizontal(
                "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
            )
            .round(2)
            .alias(self.target)
        )
        data = data.with_columns(
            pl.when(pl.col(self.target).eq(0.0))
            .then(pl.lit(False))
            .otherwise(pl.lit(True))
            .alias(self.target)
        )
        data = data.select(["id", "comment_text", "dataset", self.target])
        return data

    @typechecked
    def concat_data(self) -> pl.DataFrame:
        """This is used to concatenate the train and test data."""
        data: pl.DataFrame = self.add_target()
        test_data: pl.DataFrame = self._join_data()
        # Concatenate the data
        toxic_comments_data: pl.DataFrame = pl.concat([data, test_data], how="vertical")

        return toxic_comments_data

    @typechecked
    def prepare_data(self) -> pl.DataFrame:
        data: pl.DataFrame = self.concat_data()
        data = data.with_columns(text=pl.col("comment_text"))
        data = data.select(["text", "dataset", self.target])
        return data


class CyberBullyDataLoader(DataLoader):
    @typechecked
    def __init__(self, path: str | Path, separator: str = ",") -> None:
        """Cyber Bully DataLoader."""
        super().__init__(path, separator)

    @typechecked
    def load_data(self) -> pl.DataFrame:
        return pl.read_csv(self.path, separator=self.separator)

    @typechecked
    def dataset_name(self) -> pl.DataFrame:
        data: pl.DataFrame = self.load_data()
        data = data.with_columns(dataset=pl.lit("cyber_bully"))
        return data

    @typechecked
    def add_target(self) -> pl.DataFrame:
        data: pl.DataFrame = self.dataset_name()
        # Add target label
        data = data.with_columns(
            pl.when(pl.col("cyberbullying_type").eq("not_cyberbullying"))
            .then(pl.lit(False))
            .otherwise(pl.lit(True))
            .alias(self.target)
        )
        return data

    @typechecked
    def prepare_data(self) -> pl.DataFrame:
        data: pl.DataFrame = self.add_target()
        data = data.with_columns(text=pl.col("tweet_text"))
        data = data.select(["text", "dataset", self.target])
        return data
