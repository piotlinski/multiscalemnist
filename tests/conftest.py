import pytest
from yacs.config import CfgNode

from multiscalemnist.config import _C as cfg


@pytest.fixture
def sample_config() -> CfgNode:
    """Return sample config with default values."""
    config = cfg.clone()
    config.MIN_DIGTS = 1
    config.MAX_DIGITS = 3
    config.TRAIN_LENGTH = 10
    config.TEST_LENGTH = 5
    config.CHUNK_SIZE = 1
    return config
