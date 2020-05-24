"""MultiScaleMNIST config file."""
import logging
from pathlib import Path
from typing import Optional

from yacs.config import CfgNode

logger = logging.getLogger(__name__)


_C = CfgNode()
_C.SUBSET = "train"
_C.IMAGE_SIZE = (512, 512)
_C.DIGIT_SIZE = (128, 128)
_C.DIGIT_SCALES = (1, 2, 3)
_C.MIN_DIGTS = 2
_C.MAX_DIGITS = 5


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
