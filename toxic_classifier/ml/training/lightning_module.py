"""This module contains the LightningModel class."""

from abc import abstractmethod
from collections import defaultdict
from typing import Any

import lightning as l
import mlflow
import torch
import torchmetrics
from torch import Tensor, nn
from torch.optim import Optimizer
from transformers import BatchEncoding
from typeguard import typechecked

from toxic_classifier.ml.training.lightning_schedulers import LightningScheduler
from toxic_classifier.ml.training.loss_functions import LossFunction
from toxic_classifier.ml.transformations import Transformation
from toxic_classifier.utilities.torch_utils import plot_confusion_matrix


class BaseLightningModule(l.LightningModule):
    """This is the base class for all Lightning models."""

    def __init__(
        self,
        model: nn.Module,
        loss: LossFunction,
        optimizer: Optimizer,
        scheduler: LightningScheduler | None = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.model_size = self._calculate_model_size()

    def _calculate_model_size(self) -> float:
        """This is used to calculate the model size."""
        param_size: int = 0
        for param in self.parameters():
            param_size += param.nelement() & param.element_size()

        buffer_size: int = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() & buffer.element_size()

        size_all_mb: float = (param_size + buffer_size) / 1024**2
        return size_all_mb

    @abstractmethod
    def configure_optimizers(
        self,
    ) -> Optimizer | tuple[list[Optimizer]] | list[dict[str, Any]]:
        optimizer: Optimizer = self.optimizer(self.parameters())

        if self.scheduler is not None:
            scheduler: dict[str, Any] = self.scheduler.configure(  # type: ignore
                optimizer=optimizer,
                estimated_stepping_batches=self.trainer.estimated_stepping_batches,
            )
            return [optimizer], [scheduler]
        return optimizer

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """This is used to perform a training step."""
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """This is used to perform a validation step."""
        raise NotImplementedError


class BinaryTextClassificationLightningModule(BaseLightningModule):
    """This is used to create a Lightning Model for binary text classification."""

    @typechecked
    def __init__(
        self,
        model: nn.Module,
        loss: LossFunction,
        optimizer: Optimizer,
        scheduler: LightningScheduler | None = None,
    ) -> None:
        super().__init__(model, loss, optimizer, scheduler)

        self.training_accuracy = torchmetrics.Accuracy("binary")
        self.training_accuracy = torchmetrics.Accuracy("binary")

        self.training_roc_auc = torchmetrics.AUROC("binary")
        self.training_roc_auc = torchmetrics.AUROC("binary")

        self.training_f1_score = torchmetrics.F1Score("binary")
        self.training_f1_score = torchmetrics.F1Score("binary")

        self.train_step_outputs: dict[str, Any] = defaultdict(list)
        self.validation_step_outputs: dict[str, Any] = defaultdict(list)

        self.pos_weight: Tensor | None = None

    @typechecked
    def set_pos_weight(self, pos_weight: Tensor | None) -> None:
        """This is used to set the pos_weight."""
        self.pos_weight = pos_weight

    @typechecked
    def forward(self, encodings: BatchEncoding) -> Tensor:
        """This is used to perform a forward pass."""
        output: Tensor = self.model(encodings)
        return output

    @typechecked
    def _shared_steps(self, batch: Any) -> tuple[Tensor, ...]:
        """This is used to perform shared steps."""
        features, labels = batch

        # Forward prop
        logits: Tensor = self(features)

        # loss
        loss: LossFunction = self.loss(logits, labels, self.pos_weight)

        return loss, logits, labels

    @typechecked
    def on_train_epoch_end(self) -> None:
        """This is used to perform end of epoch operations."""
        all_logits = torch.stack(self.train_step_outputs["logits"])
        all_labels = torch.stack(self.train_step_outputs["labels"])

        confusion_matrix = self.training_confusion_matrix(all_logits, all_labels)
        figure = plot_confusion_matrix(confusion_matrix, ["0", "1"])
        mlflow.log_figure(figure, "training_confusion_matrix.png")

        self.train_step_outputs = defaultdict(list)

    @typechecked
    def on_validation_epoch_end(self) -> None:
        """This is used to perform end of epoch operations."""
        all_logits = torch.stack(self.validation_step_outputs["logits"])
        all_labels = torch.stack(self.validation_step_outputs["labels"])

        confusion_matrix = self.validation_confusion_matrix(all_logits, all_labels)
        figure = plot_confusion_matrix(confusion_matrix, ["0", "1"])
        mlflow.log_figure(figure, "training_confusion_matrix.png")

        self.validation_step_outputs = defaultdict(list)

    @typechecked
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        # Compute the loss, logits, and labels
        loss, logits, labels = self._shared_steps(batch)
        self.log("training_loss", loss)

        # Accuracy
        self.training_accuracy(logits, labels)
        self.log(
            "training_accuracy",
            self.training_accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        # ROC AUC
        self.training_roc_auc(logits, labels)
        self.log(
            "training_roc_auc",
            self.training_roc_auc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        # F1 Score
        self.training_f1_score(logits, labels)
        self.log(
            "training_f1_score",
            self.training_f1_score,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        self.train_step_outputs["logits"].append(logits)
        self.train_step_outputs["labels"].append(labels)

        return loss

    @typechecked
    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        # Compute the loss, logits, and labels
        loss, logits, labels = self._shared_steps(batch)
        self.log("validation_loss", loss)

        # Accuracy
        self.validation_accuracy(logits, labels)
        self.log(
            "validation_accuracy",
            self.validation_accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        # ROC AUC
        self.validation_roc_auc(logits, labels)
        self.log(
            "validation_roc_auc",
            self.validation_roc_auc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        # F1 Score
        self.validation_f1_score(logits, labels)
        self.log(
            "validation_f1_score",
            self.validation_f1_score,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        self.validation_step_outputs["logits"].append(logits)
        self.validation_step_outputs["labels"].append(labels)

        return {"loss": loss, "predictions": logits, "labels": labels}

    @typechecked
    def get_transformation(self) -> Transformation:
        return self.model.get_transformation()
