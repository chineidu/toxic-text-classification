"""This is used to load the transformations (tokenizers, etc)"""

import os
from abc import ABC, abstractmethod

from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
)
from typeguard import typechecked

from toxic_classifier.utilities.utils_io import (
    is_dir,
    is_file,
    translate_gcs_dir_to_local,
)


class Transformation(ABC):
    @abstractmethod
    def __init__(self, save_directory: str) -> None:
        """This is used to initialize the transformation."""
        self.save_directory = save_directory

    @abstractmethod
    def __call__(self, texts: list[str]) -> BatchEncoding:
        """This is used to encode the texts."""
        raise NotImplementedError()


class HuggingFaceTransformation(Transformation):
    @typechecked
    def __init__(
        self,
        save_directory: str,
        truncation: bool = True,
        padding: bool = True,
        max_length: int = 80,
    ) -> None:
        super().__init__(save_directory)
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length

    @typechecked
    def init_tokenizer(self) -> PreTrainedTokenizerBase:
        """This is used to load the pretrained tokenizer."""
        if is_dir(self.save_directory):
            tokenizer_dir: str = translate_gcs_dir_to_local(self.save_directory)
        elif is_file(self.save_directory):
            pretrained_tokenizer_name_or_path = translate_gcs_dir_to_local(self.save_directory)
            tokenizer_dir = os.path.dirname(pretrained_tokenizer_name_or_path)
        else:
            tokenizer_dir = self.save_directory

        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_dir
        )
        return tokenizer

    @typechecked
    def __call__(self, texts: list[str]) -> BatchEncoding:
        tokenizer: PreTrainedTokenizerBase = self.init_tokenizer()
        output: BatchEncoding = tokenizer(
            texts,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_length,
        )
        return output
