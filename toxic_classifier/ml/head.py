from typing import Any

from torch import Tensor, nn
from typeguard import typechecked


class Head(nn.Module):
    """Base class for all heads. It's used to defined tasks such as
    classification or regression."""

    pass


class BinaryClassifierHead(Head):
    @typechecked
    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the binary classification head module."""
        super().__init__()

        self.head = nn.Sequential(nn.Linear(in_features, out_features), nn.Sigmoid())

    @typechecked
    def forward(self, features: Tensor | Any) -> Tensor:
        """Forward pass of the binary classification head."""
        output: Tensor = self.head(features)
        return output


class SoftmaxClassifierHead(Head):
    @typechecked
    def __init__(self, in_features: int, out_features: int, dim: int = 1) -> None:
        """Initialize the softmax classification head module."""
        super().__init__()

        self.head = nn.Sequential(nn.Linear(in_features, out_features), nn.Softmax(dim=dim))

    @typechecked
    def forward(self, features: Tensor) -> Tensor:
        """Forward pass of the softmax classification head."""
        output: Tensor = self.head(features)
        return output
