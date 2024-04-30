from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
from typeguard import typechecked


class TextDataset(Dataset):
    """This is the base class for all Datasets."""

    @typechecked
    def __init__(
        self,
        text_features: list[str],
        labels: npt.NDArray[np.float_],
    ) -> None:
        super().__init__()

        self.features = text_features
        self.labels = torch.tensor(labels, dtype=torch.float)

    @typechecked
    def __getitem__(self, index: int | Any) -> tuple[torch.Tensor | Any, torch.Tensor]:
        feature = self.features[index]
        label = self.labels[index]
        return feature, label

    @typechecked
    def __len__(self) -> int:
        return len(self.features)
