import torch.nn.functional as F
from torch import Tensor, nn
from typeguard import typechecked


class Head(nn.Module):
    pass


class BinaryClassifierHead(Head):
    @typechecked
    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the binary classification head module."""
        self.head = nn.Sequential(nn.Linear(in_features, out_features), F.sigmoid())

    @typechecked
    def forward(self, features: Tensor) -> Tensor:
        """Forward pass of the binary classification head."""
        output: Tensor = self.head(features)
        return output


class SoftmaxClassifierHead(Head):
    @typechecked
    def __init__(self, in_features: int, out_features: int, dim: int = 1) -> None:
        """Initialize the softmax classification head module."""
        self.head = nn.Sequential(nn.Linear(in_features, out_features), F.softmax(dim=dim))

    @typechecked
    def forward(self, features: Tensor) -> Tensor:
        """Forward pass of the softmax classification head."""
        output: Tensor = self.head(features)
        return output
