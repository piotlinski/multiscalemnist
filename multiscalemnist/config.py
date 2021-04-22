"""MultiScaleMNIST config file."""
import logging
from pathlib import Path
from typing import Optional

from yacs.config import CfgNode

logger = logging.getLogger(__name__)


_C = CfgNode()
_C.DIRECTORY = "dataset"
_C.IMAGE_SIZE = (512, 512)
_C.N_CHANNELS = 3
_C.GRID_SIZES = (3, 3), (2, 2)
_C.MIN_DIGIT_SIZE = 64
_C.MAX_DIGIT_SIZE = 128
_C.POSITION_VARIANCE = 0.6
_C.CELL_FILLED_THRESHOLD = 0.4
_C.TRAIN_LENGTH = 40_000
_C.TEST_LENGTH = 4_000
_C.CHUNK_SIZE = 32
_C.DIGIT_SET = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
_C.SEED = 0


def get_config(config_file: Optional[str] = None, **kwargs) -> CfgNode:
    """Get yacs config with default values."""
    config = _C.clone()
    if config_file is not None:
        config_path = Path(config_file)
        if config_path.exists():
            config.merge_from_file(config_file)
        else:
            logger.warning("File %s does not exist.", config_file)
    config.update(**kwargs)
    config.freeze()
    return config
