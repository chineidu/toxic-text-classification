from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
from typeguard import typechecked


class TextDataset(Dataset):
    """This is the base class for all Datasets."""

    # @typechecked
    def __init__(
        self,
        features: npt.NDArray[np.float_],
        labels: npt.NDArray[np.float_],
    ) -> None:
        super().__init__()

        try:
            self.features = torch.tensor(features, dtype=torch.float32)
        except TypeError:
            self.features = features
        self.labels = torch.tensor(labels, dtype=torch.long)

    @typechecked
    def __getitem__(self, index: int) -> tuple[torch.Tensor | Any, torch.Tensor]:
        feature = self.features[index]
        label = self.labels[index]
        return feature, label

    @typechecked
    def __len__(self) -> int:
        return len(self.features)
