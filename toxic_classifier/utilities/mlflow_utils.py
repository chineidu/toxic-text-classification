"""This module contains the utility MLFlow functions."""

from contextlib import contextmanager
from typing import Iterable

import mlflow
from mlflow import ActiveRun
from typeguard import typechecked


@typechecked
def create_experiment(experiment_name: str | None = None) -> None:
    """This is used to create and set an MLFlow experiment."""
    if experiment_name is None:
        experiment_name = "Default"

    try:
        mlflow.create_experiment(experiment_name=experiment_name)
    except mlflow.exceptions.RestException:
        pass
    finally:
        mlflow.set_experiment(experiment_name=experiment_name)


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
