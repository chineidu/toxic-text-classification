import pkg_resources  # type: ignore
import polars as pl
from spacy.lang.en.stop_words import STOP_WORDS
from symspellpy import SymSpell
from typeguard import typechecked


class SpellChecker:
    def __init__(self, max_dictionary_edit_distance: int = 2, prefix_length: int = 7) -> None:
        """SymSpellChecker constructor. It uses SymSpellPy to spell check the data."""

        self.max_dictionary_edit_distance = max_dictionary_edit_distance
        self.prefix_length = prefix_length
        self.dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        self.bigram_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
        )

    def _load_sym_spell_model(self) -> SymSpell:
        """This loads the sym spell model."""
        sym_spell = SymSpell(
            max_dictionary_edit_distance=self.max_dictionary_edit_distance,
            prefix_length=self.prefix_length,
        )
        sym_spell.load_dictionary(self.dictionary_path, term_index=0, count_index=1)
        sym_spell.load_bigram_dictionary(self.bigram_path, term_index=0, count_index=2)
        return sym_spell

    def __call__(self, *args, **kwargs) -> str:
        sym_spell = self._load_sym_spell_model()
        return sym_spell.lookup_compound(max_edit_distance=2, *args, **kwargs)[0].term


class TextDataCleaner:
    def __init__(self, data: pl.DataFrame) -> None:
        """TextDataCleaner constructor."""
        self.data = data

    @typechecked
    def remove_punctuation(self) -> pl.DataFrame:
        """This is used to remove punctuation from the data."""
        data: pl.DataFrame = self.data.with_columns(
            text=pl.col("text").str.replace_all(pattern=r"[^\w\s]", value="")
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
            text=pl.col("text").str.to_lowercase().map_elements(self._remove_stopwords)
        )
        return data

    @typechecked
    def remove_non_letters(self, data: pl.DataFrame) -> pl.DataFrame:
        """This is used to remove non-letters from the data."""
        data = data.with_columns(text=pl.col("text").str.replace_all(pattern=r"[0-9]", value=""))
        return data

    @typechecked
    def remove_urls(self, data: pl.DataFrame) -> pl.DataFrame:
        """This is used to remove urls from the data."""
        data = data.with_columns(
            text=pl.col("text").str.replace_all(pattern=r"https?://\S{1,150}", value="")
        )
        return data

    @typechecked
    def _spell_correct(self, phrase: str) -> str:
        """This is used to spell correct the phrase."""
        my_spell_checker = SpellChecker()
        cleaned_text: str = my_spell_checker(phrase=phrase)
        return cleaned_text

    @typechecked
    def spell_correct(self, data: pl.DataFrame) -> pl.DataFrame:
        """This is used to spell correct the data.

        NB:
        This is a very expensive operation.
        """
        data = data.with_columns(text=pl.col("text").map_elements(self._spell_correct))
        return data

    @typechecked
    def prepare_data(self) -> pl.DataFrame:
        """This is used to apply the all the data cleaning steps to the data."""
        data: pl.DataFrame = self.remove_punctuation()
        data = self.lowercase_and_filter_stopwords(data)
        data = self.remove_non_letters(data)
        data = self.remove_urls(data)
        return data
