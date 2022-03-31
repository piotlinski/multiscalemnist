"""Test SSD config."""
from unittest.mock import patch

from multiscalemnist.config import get_config


def test_default_config():
    """Test if default config is fetched."""
    config = get_config()
    assert config is not None


@patch("multiscalemnist.config.Path.exists", return_value=True)
@patch("multiscalemnist.config.CfgNode.merge_from_file")
def test_updating_with_yaml(merge_mock, _exists_mock):
    """Test getting config and updating with yaml."""
    config = get_config(config_file="test")
    merge_mock.assert_called_with("test")
    assert config is not None


@patch("multiscalemnist.config.Path.exists", return_value=False)
@patch("multiscalemnist.config.CfgNode.merge_from_file")
def test_no_update_when_no_file(merge_mock, _exists_mock):
    """Test if no config update occurs when file does not exist."""
    config = get_config(config_file="test")
    merge_mock.assert_not_called()
    assert config is not None


def test_updating_with_kwargs():
    """Test updating config with additional kwargs."""
    kwargs = {"test": 1, "another_test": 2}
    config = get_config(**kwargs)
    assert config is not None
    assert kwargs.items() <= config.items()
