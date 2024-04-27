"""This is used to define schedulers.

Docs: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import typechecked


class LightningScheduler(ABC):
    """Abstract base class for all schedulers."""

    def __init__(
        self,
        scheduler: _LRScheduler,
        interval: Literal["step", "epoch"] = "epoch",
        frequency: int = 1,
        monitor: str = "val_loss",
        strict: bool = True,
        name: str | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor
        self.strict = strict
        self.name = name

    @abstractmethod
    def configure_scheduler(self) -> dict[str, Any]:
        raise NotImplementedError


class LightningLRScheduler(LightningScheduler):
    """This is used to create a Lightning Scheduler."""

    @typechecked
    def configure_scheduler(self, optimizer: Optimizer) -> dict[str, Any]:
        """This is used to configure the scheduler."""
        return {
            "scheduler": self.scheduler(optimizer),
            "interval": self.interval,
            "frequency": self.frequency,
            "monitor": self.monitor,
            "strict": self.strict,
            "name": self.name,
        }
