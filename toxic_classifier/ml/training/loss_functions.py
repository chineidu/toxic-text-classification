"""This module is used to define the loss functions."""

from typing import Literal

import torch.nn.functional as F
from torch import Tensor, nn
from typeguard import typechecked


class LossFunction(nn.Module):
    pass


class BinaryCrossEntropyLoss(LossFunction):
    @typechecked
    def __init__(self, reduction: Literal["none", "mean", "sum"] = "none") -> None:
        super().__init__()
        self.reduction = reduction
        self.criterion = F.binary_cross_entropy_with_logits()

    @typechecked
    def forward(self, input: Tensor, target: Tensor, pos_weight: Tensor | None = None) -> Tensor:
        loss: Tensor = self.criterion(
            input, target, reduction=self.reduction, pos_weight=pos_weight
        )
        return loss
