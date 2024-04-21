import polars as pl
from spacy.lang.en.stop_words import STOP_WORDS
from typeguard import typechecked


class DataCleaner:
    def __init__(self, data: pl.DataFrame) -> None:
        """DataCleaner constructor."""
        self.data = data

    @typechecked
    @staticmethod
    def remove_punctuation(data: pl.DataFrame) -> pl.DataFrame:
        """This is used to remove punctuation from the data."""
        data = data.with_columns(
            cleaned_text=pl.col("cleaned_text").str.replace_all(pattern=r"[^\w\s]", value="")
        )
        return data

    @typechecked
    @staticmethod
    def _remove_stopwords(text: str) -> str:
        """This is used to remove stopwords from the text."""
        words: list[str] = [word for word in text.split() if word not in STOP_WORDS]
        return " ".join(words)

    @typechecked
    def lowercase_and_filter_stopwords(self, data: pl.DataFrame) -> pl.DataFrame:
        """This is used to lowercase and filter stopwords from the data."""
        data = data.with_columns(
            cleaned_text=pl.col("cleaned_text")
            .str.to_lowercase()
            .map_elements(self._remove_stopwords, return_dtype=pl.Utf8)
        )
        return data

    @typechecked
    def remove_non_letters(self, data: pl.DataFrame) -> pl.DataFrame:
        """This is used to remove non-letters from the data."""
        data = data.with_columns(
            cleaned_text=pl.col("cleaned_text").str.replace_all(pattern=r"[0-9]", value="")
        )
        return data

    @typechecked
    def remove_urls(self, data: pl.DataFrame) -> pl.DataFrame:
        """This is used to remove urls from the data."""
        data = data.with_columns(
            cleaned_text=pl.col("cleaned_text").str.replace_all(
                pattern=r"https?://\S{1,150}", value=""
            )
        )
        return data

    @typechecked
    def _spell_correct(self, phrase: str) -> str:
        """This is used to spell correct the phrase."""
        my_spell_checker = SpellChecker()  # type: ignore
        cleaned_text: str = my_spell_checker(phrase=phrase)
        return cleaned_text

    @typechecked
    def spell_correct(self, data: pl.DataFrame) -> pl.DataFrame:
        """This is used to spell correct the data.

        NB:
        This is a very expensive operation.
        """
        data = data.with_columns(
            text=pl.col("cleaned_text").map_elements(self._spell_correct, return_dtype=pl.Utf8)
        )
        return data

    @typechecked
    def prepare_data(self) -> pl.DataFrame:
        """This is used to apply the all the data cleaning steps to the data."""
        data: pl.DataFrame = self.data.with_columns(cleaned_text=pl.col("text"))
        data = self.remove_urls(data)
        data = self.remove_punctuation(data)
        data = self.lowercase_and_filter_stopwords(data)
        data = self.remove_non_letters(data)
        return data
