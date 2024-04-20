import polars as pl
from typeguard import typechecked

from toxic_classifier import DATA_PATH, config
from toxic_classifier.utilities.datacleaners import DataCleaner
from toxic_classifier.utilities.dataloaders import (
    CyberBullyDataLoader,
    GHCDataLoader,
    ToxicCommentsDataLoader,
)


class PrepareData:
    @typechecked
    @staticmethod
    def _prepare_gabe_hate_corpus_data() -> pl.DataFrame:
        """This is used to prepare the Gabe Hate Corpus data."""
        dataloader: GHCDataLoader = GHCDataLoader(
            path=DATA_PATH / config.data_paths.gabe_hate_corpus.path,
            separator=config.data_paths.gabe_hate_corpus.separator,
        )
        data: pl.DataFrame = dataloader.prepare_data()
        return data

    @typechecked
    @staticmethod
    def _prepare_cyberbully_data() -> pl.DataFrame:
        """This is used to prepare the Cyberbully data."""
        dataloader: CyberBullyDataLoader = CyberBullyDataLoader(
            path=DATA_PATH / config.data_paths.cyberbully.path,
            separator=config.data_paths.cyberbully.separator,
        )
        data: pl.DataFrame = dataloader.prepare_data()
        return data

    @typechecked
    @staticmethod
    def _prepare_toxic_comments_data() -> pl.DataFrame:
        """This is used to prepare the Toxic Comments data."""
        dataloader: ToxicCommentsDataLoader = ToxicCommentsDataLoader(
            path=DATA_PATH / config.data_paths.toxic_comments.path,
            separator=config.data_paths.toxic_comments.separator,
            other_path=DATA_PATH / config.data_paths.toxic_comments.other_path,
            labels_path=DATA_PATH / config.data_paths.toxic_comments.labels_path,
        )
        data: pl.DataFrame = dataloader.prepare_data()
        return data

    @typechecked
    def concat_data(self) -> pl.DataFrame:
        """This is used to prepare the data."""
        gabe_hate_corpus_data: pl.DataFrame = self._prepare_gabe_hate_corpus_data()
        cyberbully_data: pl.DataFrame = self._prepare_cyberbully_data()
        toxic_comments_data: pl.DataFrame = self._prepare_toxic_comments_data()
        data: pl.DataFrame = pl.concat(
            [gabe_hate_corpus_data, cyberbully_data, toxic_comments_data],
            how="vertical",
        )
        return data

    @typechecked
    def clean_data(self) -> pl.DataFrame:
        """This is used to clean the data."""
        data: pl.DataFrame = self.concat_data()
        cleaner: DataCleaner = DataCleaner(data)
        data = cleaner.prepare_data()
        return data
