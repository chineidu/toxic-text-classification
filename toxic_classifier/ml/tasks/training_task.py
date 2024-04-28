"""T"""

from abc import ABC, abstractmethod

from lightning import Trainer
from omegaconf import DictConfig
from typeguard import typechecked

from toxic_classifier.datamodule.datamodule import BaseDataModule
from toxic_classifier.ml.training.lightning_module import BaseLightningModule
from toxic_classifier.utilities.logger import get_rich_logger
from toxic_classifier.utilities.mlflow_utils import activate_mlflow
from toxic_classifier.utilities.utils_io import is_file


class BaseTrainingTask(ABC):
    """This is the base class for all training tasks."""

    @abstractmethod
    def __init__(
        self,
        name: str,
        datamodule: BaseDataModule,
        lightning_module: BaseLightningModule,
        trainer: Trainer,
        best_training_checkpoint_path: str,
        last_training_checkpoint_path: str,
    ):
        self.name = name
        self.datamodule = datamodule
        self.lightning_module = lightning_module
        self.trainer = trainer
        self.best_training_checkpoint_path = best_training_checkpoint_path
        self.last_training_checkpoint_path = last_training_checkpoint_path
        self.logger = get_rich_logger(name=self.__class__.__name__)

    @abstractmethod
    def run(self, config: DictConfig) -> None:
        raise NotImplementedError()


class TrainingTask(BaseTrainingTask):
    """This is used for training the model."""

    @typechecked
    def __init__(
        self,
        name: str,
        datamodule: BaseDataModule,
        lightning_module: BaseLightningModule,
        trainer: Trainer,
        best_training_checkpoint_path: str,
        last_training_checkpoint_path: str,
    ):
        super().__init__(
            name,
            datamodule,
            lightning_module,
            trainer,
            best_training_checkpoint_path,
            last_training_checkpoint_path,
        )

    @typechecked
    def run(self, config: DictConfig) -> None:
        """This is used to run the training task."""
        experiment_name: str = config.mlflow.experiment_name
        run_id: str = config.mlflow.run_id
        run_name: str = config.mlflow.run_name

        with activate_mlflow(
            experiment_name=experiment_name, run_id=run_id, run_name=run_name
        ) as _:
            if is_file(self.last_training_checkpoint_path):
                self.logger.info(
                    (
                        f"Found checkpoint here: {self.last_training_checkpoint_path}. "
                        "Resuming training..."
                    )
                )
                self.trainer.fit(
                    model=self.lightning_module,
                    datamodule=self.datamodule,
                    ckpt_path=self.last_training_checkpoint_path,
                )
            else:
                self.trainer.fit(model=self.lightning_module, datamodule=self.datamodule)
            self.logger.info("Training finished.")
