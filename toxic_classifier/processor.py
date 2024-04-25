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
    @staticmethod
    def convert_target_columns(data: pl.DataFrame) -> pl.DataFrame:
        """This is used to convert target columns to numeric."""
        target: str = config.constants.target
        data = data.with_columns(
            pl.when(pl.col(target).eq(True)).then(pl.lit(1)).otherwise(pl.lit(0)).alias(target)
        )
        return data

    @typechecked
    def clean_data(self) -> pl.DataFrame:
        """This is used to clean the data."""
        data: pl.DataFrame = self.concat_data()
        data = self.convert_target_columns(data)
        cleaner: DataCleaner = DataCleaner(data)
        data = cleaner.prepare_data().sample(fraction=1, seed=config.constants.seed)
        # Re-arrange the columns
        data = data.select(config.data_cleaner.column_names)
        # Remove texts with a single word
        data = data.with_columns(length=pl.col("cleaned_text").str.len_chars())
        data = data.filter(pl.col("length").gt(1)).drop("length")

        return data
