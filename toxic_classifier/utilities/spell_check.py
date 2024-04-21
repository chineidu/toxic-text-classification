import pkg_resources  # type: ignore
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

    @typechecked
    def _load_sym_spell_model(self) -> SymSpell:
        """This loads the sym spell model."""
        sym_spell = SymSpell(
            max_dictionary_edit_distance=self.max_dictionary_edit_distance,
            prefix_length=self.prefix_length,
        )
        sym_spell.load_dictionary(self.dictionary_path, term_index=0, count_index=1)
        sym_spell.load_bigram_dictionary(self.bigram_path, term_index=0, count_index=2)
        return sym_spell

    @typechecked
    def __call__(self, *args, **kwargs) -> str:
        sym_spell = self._load_sym_spell_model()
        return sym_spell.lookup_compound(max_edit_distance=2, *args, **kwargs)[0].term
