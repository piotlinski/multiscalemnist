"""Test original MNIST handlers."""
from pathlib import Path
from unittest.mock import call, patch

from multiscalemnist.mnist import download_mnist, verify_mnist_dir


@patch("multiscalemnist.mnist.download_mnist")
@patch("multiscalemnist.mnist.Path.mkdir")
@patch("multiscalemnist.mnist.Path.exists", return_value=False)
def test_mnist_dir_verification(_exists_mock, mkdir_mock, download_mock):
    """Test if MNIST data dir is verified."""
    verify_mnist_dir(data_dir=Path("test"), mnist_keys=["test"])
    mkdir_mock.assert_called()
    download_mock.assert_called()


@patch("multiscalemnist.mnist.subprocess.call")
def test_download_mnist(call_mock):
    """Test downloading MNIST dataset."""
    data_dir = Path("test")
    mnist_keys = ["test"]
    mnist_url = "http://test.url/"
    target_path = data_dir.joinpath(mnist_keys[0] + ".gz")
    download_mnist(data_dir=data_dir, mnist_keys=mnist_keys, mnist_url=mnist_url)
    call_mock.assert_has_calls(
        [
            call(f"curl {mnist_url}{mnist_keys[0]}.gz -o {str(target_path)}"),
            call(f"gunzip -d {str(target_path)}"),
        ]
    )
