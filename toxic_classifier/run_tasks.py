from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import torch
from lightning import Trainer, seed_everything
from omegaconf import DictConfig

from toxic_classifier import DATA_PATH, SRC_ROOT, config
from toxic_classifier.datamodule.datamodule import TextDataModule
from toxic_classifier.ml.backbone import BaseBackbone, HuggingFaceBackbone
from toxic_classifier.ml.head import BinaryClassifierHead
from toxic_classifier.ml.models import BinaryTextClassifier
from toxic_classifier.ml.tasks.training_task import TrainingTask
from toxic_classifier.ml.training.lightning_module import (
    BinaryTextClassificationLightningModule,
)
from toxic_classifier.ml.training.lightning_schedulers import (
    BaseLightningScheduler,
    LightningLRScheduler,
)
from toxic_classifier.ml.training.loss_functions import (
    BaseLossFunction,
    BinaryCrossEntropyLoss,
)
from toxic_classifier.ml.transformations import (
    BaseTransformation,
    HuggingFaceTokenizationTransformation,
)
from toxic_classifier.utilities.logger import get_rich_logger
from toxic_classifier.utilities.torch_utils import get_local_rank


# @hydra.main(config_path="./config/", config_name="config", version_base=None)
def run_tasks(*, config: DictConfig, task: TrainingTask) -> None:
    logger = get_rich_logger(__file__)
    # assert config.mlflow.run_id is not None, "Run id has to be set for running tasks"

    backend: str = "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{get_local_rank()}")
        backend = "nccl"

    torch.distributed.init_process_group(backend=backend)

    seed_everything(seed=config.constants.seed, workers=True)
    logger.info("Running task")
    task.run(config=config)


if __name__ == "__main__":
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    data: pl.DataFrame = pl.read_parquet(DATA_PATH / "cleaned_data/cleaned_data.parquet")
    features: list[str] = list(data.select(["cleaned_text"]).to_numpy().squeeze())
    labels: npt.NDArray[np.float_] = data.select([config.constants.target]).to_numpy().squeeze()

    transformation: BaseTransformation = HuggingFaceTokenizationTransformation(
        save_directory=f"{SRC_ROOT.parent}/{config.models.transformations.save_directory}"
    )
    dm: TextDataModule = TextDataModule(
        text_features=features,
        labels=labels,
        batch_size=64,
        transformation=transformation,
    )

    backbone: BaseBackbone = HuggingFaceBackbone(
        transformation=transformation,
        pretrained_model_name_or_path=config.models.backbone.save_directory,
        pretrained=True,
    )
    head: BinaryClassifierHead = BinaryClassifierHead(in_features=128, out_features=1)
    model = BinaryTextClassifier(backbone=backbone, head=head, adapter=None)
    loss_func: BaseLossFunction = BinaryCrossEntropyLoss()
    scheduler: Any = ReduceLROnPlateau
    lightning_scheduler: BaseLightningScheduler = LightningLRScheduler(scheduler=scheduler)
    lightning_module: BinaryTextClassificationLightningModule = (
        BinaryTextClassificationLightningModule(
            model=model, loss=loss_func, optimizer=AdamW, scheduler=None
        )
    )

    trainer = Trainer(
        fast_dev_run=2,
        max_epochs=10,
        accelerator="auto",  # set to "auto" or "gpu" to use GPUs if available
        devices="auto",  # Uses all available GPUs if applicable
    )
    task: TrainingTask = TrainingTask(
        name="training",
        datamodule=dm,
        lightning_module=lightning_module,
        trainer=trainer,
        best_training_checkpoint_path=None,
        last_training_checkpoint_path=None,
    )

    run_tasks(config=config, task=task)
