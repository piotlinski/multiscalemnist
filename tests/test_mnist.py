"""Test original MNIST handlers."""
from pathlib import Path
from unittest.mock import call, patch

import numpy as np

from multiscalemnist.mnist import (
    download_mnist,
    fetch_mnist,
    load_images,
    load_labels,
    verify_mnist_dir,
)


@patch("multiscalemnist.mnist.download_mnist")
@patch("multiscalemnist.mnist.Path.mkdir")
@patch("multiscalemnist.mnist.Path.exists", return_value=False)
def test_mnist_dir_verification(_exists_mock, mkdir_mock, download_mock):
    """Test if MNIST data dir is verified."""
    verify_mnist_dir(data_dir=Path("test"), mnist_keys=("test",))
    mkdir_mock.assert_called()
    download_mock.assert_called()


@patch("multiscalemnist.mnist.subprocess.call")
def test_download_mnist(call_mock):
    """Test downloading MNIST dataset."""
    data_dir = Path("test")
    mnist_keys = ("test",)
    mnist_url = "http://test.url/"
    target_path = data_dir.joinpath(mnist_keys[0] + ".gz")
    download_mnist(data_dir=data_dir, mnist_keys=mnist_keys, mnist_url=mnist_url)
    call_mock.assert_has_calls(
        [
            call(f"curl {mnist_url}{mnist_keys[0]}.gz -o {str(target_path)}"),
            call(f"gunzip -d {str(target_path)}"),
        ]
    )


@patch(
    "multiscalemnist.mnist.np.fromfile", return_value=np.random.random(16 + 3 * 28 * 28)
)
@patch("multiscalemnist.mnist.Path.open")
def test_load_images(open_mock, from_file_mock):
    """Test loading images from file."""
    loaded = load_images(data_dir=Path("test"), images_file="test_file")
    from_file_mock.assert_called_with(
        file=open_mock().__enter__.return_value, dtype=np.uint8
    )
    assert loaded.shape == (3, 28, 28, 1)


@patch("multiscalemnist.mnist.np.fromfile", return_value=np.random.random(8 + 3))
@patch("multiscalemnist.mnist.Path.open")
def test_load_labels(open_mock, from_file_mock):
    """Test loading labels from file."""
    loaded = load_labels(data_dir=Path("test"), labels_file="test_file")
    from_file_mock.assert_called_with(
        file=open_mock().__enter__.return_value, dtype=np.uint8
    )
    assert loaded.shape == (3,)


@patch("multiscalemnist.mnist.verify_mnist_dir")
@patch("multiscalemnist.mnist.load_labels")
@patch("multiscalemnist.mnist.load_images")
def test_fetching_mnist(load_images_mock, load_labels_mock, _verify_mock):
    """"""
    images = [
        np.random.randint(0, 256, (3, 28, 28, 1)),
        np.random.randint(0, 255, (2, 28, 28, 1)),
    ]
    labels = [np.random.randint(0, 10, (3,)), np.random.randint(0, 10, (2,))]
    load_images_mock.side_effect = images
    load_labels_mock.side_effect = labels
    data = fetch_mnist(
        data_dir="test",
        mnist_keys=("train-images", "train-labels", "t10k-images", "t10k-labels"),
    )
    assert "train" in data
    assert "test" in data
    assert data["train"][0].shape == (3, 28, 28, 1)
    assert data["test"][0].shape == (2, 28, 28, 1)
    assert data["train"][1].shape == (3,)
    assert data["test"][1].shape == (2,)
