from pathlib import Path

import polars as pl
from omegaconf import DictConfig, OmegaConf
from typeguard import typechecked

import toxic_classifier

SRC_ROOT: Path = Path(toxic_classifier.__file__).absolute().parent  # src/
CONFIG_PATH: Path = SRC_ROOT / "utilities/config/config.yaml"

config: DictConfig = OmegaConf.load(CONFIG_PATH)


@typechecked
def _check_column_names(data: pl.DataFrame) -> pl.DataFrame:
    """Check if the column names are in the data."""
    column_names: set[str] = set(config.data_cleaner.column_names)

    for column_name in column_names:
        if column_name not in data.columns:
            raise ValueError(f"{column_name} not in data.")
        return data
