"""Test command line interface."""
from unittest.mock import patch

from click.testing import CliRunner

from multiscalemnist.cli import generate


@patch("multiscalemnist.cli.fetch_mnist")
def test_failed_command_exit_code(fetch_mock, sample_config):
    """Test if raising unhandled exception return exit code 1"""
    runner = CliRunner()
    exception = RuntimeError("Random error")

    fetch_mock.side_effect = exception

    result = runner.invoke(generate, obj=sample_config)

    assert result.exit_code == 1
    assert result.exception == exception
