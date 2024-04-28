"""This module contains the utility MLFlow functions."""

from contextlib import contextmanager
from typing import Any, Iterable

import mlflow
from mlflow import ActiveRun
from mlflow.tracking import MlflowClient
from typeguard import typechecked

from toxic_classifier.utilities.logger import get_rich_logger

logger = get_rich_logger(name=__file__)


@typechecked
def create_experiment(experiment_name: str | None = None) -> None:
    """This is used to create and set an MLFlow experiment."""
    if experiment_name is None:
        experiment_name = "Default"

    try:
        # Check if experiment already exists
        experiment_id: Any = mlflow.get_experiment_by_name(name=experiment_name)
        if experiment_id is not None:
            # If the experiment exists, set it as the current experiment
            mlflow.set_experiment(experiment_name=experiment_name)
        else:
            mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.RestException as err:
        logger.info(f"{err}")


@contextmanager  # type: ignore
def activate_mlflow(
    experiment_name: str | None = None,
    run_id: str | None = None,
    run_name: str | None = None,
) -> Iterable[ActiveRun]:
    """This is used to activate MLFlow."""
    create_experiment(experiment_name=experiment_name)

    run: ActiveRun
    with mlflow.start_run(run_id=run_id, run_name=run_name) as run:
        yield run


def get_run_id(experiment_name: str) -> str | None:
    """This is used to get the MLFlow run id."""
    client: MlflowClient = MlflowClient()
    experiment_id: str = mlflow.get_experiment_by_name(experiment_name).experiment_id
    runs = client.search_runs(
        experiment_ids=[experiment_id], max_results=1, order_by=["start_time DESC"]
    )
    if runs:
        last_run = runs[0]
        run_id: str | None = last_run.info.run_id
    else:
        run_id = None
    return run_id
