import polars as pl
from sklearn.model_selection import train_test_split
from typeguard import typechecked

from toxic_classifier import config


@typechecked
def _check_column_names(data: pl.DataFrame) -> pl.DataFrame:
    """Check if the column names are in the data."""
    column_names: set[str] = set(config.data_cleaner.column_names)

    for column_name in column_names:
        if column_name not in data.columns:
            raise ValueError(f"{column_name} not in data.")
        return data


@typechecked
def assign_split(
    data: pl.DataFrame,
    target: str = config.assign_split.target,
    stratify: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """This is used to split the data into train, test, validation, and holdout sets."""

    seed: int = config.assign_split.seed
    test_size: float = config.assign_split.test_size

    X: pl.DataFrame = data.drop([target])
    y: pl.DataFrame = data.select([target])

    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return X_train, X_test, y_train, y_test
