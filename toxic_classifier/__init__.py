from pathlib import Path

from omegaconf import DictConfig, OmegaConf

import toxic_classifier

SRC_ROOT: Path = Path(toxic_classifier.__file__).absolute().parent  # src/
CONFIG_PATH: Path = SRC_ROOT / "config/config.yaml"
DATA_PATH: Path = SRC_ROOT.parent / "data"

config: DictConfig = OmegaConf.load(CONFIG_PATH)
