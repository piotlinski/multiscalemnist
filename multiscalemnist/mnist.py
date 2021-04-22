"""Original MNIST handlers."""
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

MNIST_URL = "http://yann.lecun.com/exdb/mnist/"
MNIST_KEYS: Tuple[str, ...] = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


def verify_mnist_dir(data_dir: Path, mnist_keys: Tuple[str, ...]):
    """Check if data already downloaded and invoke downloading if needed."""
    if not all([data_dir.joinpath(file).exists() for file in mnist_keys]):
        data_dir.mkdir(exist_ok=True)
        download_mnist(data_dir=data_dir, mnist_keys=mnist_keys, mnist_url=MNIST_URL)


def download_mnist(data_dir: Path, mnist_keys: Tuple[str, ...], mnist_url: str):
    """Download MNIST dataset."""
    for key in mnist_keys:
        key += ".gz"
        url = (mnist_url + key).format(**locals())
        target_path = data_dir.joinpath(key)
        cmd = f"curl {url} -o {str(target_path)}"
        subprocess.call(cmd, shell=True)
        cmd = f"gunzip -d {str(target_path)}"
        subprocess.call(cmd, shell=True)


def load_images(data_dir: Path, images_file: str) -> np.ndarray:
    """Load data from image file.

    :param data_dir: directory contining data files
    :param images_file: mnist images file
    :return: numpy array of shape [length, 28, 28]
    """
    with data_dir.joinpath(images_file).open() as handle:
        loaded = np.fromfile(file=handle, dtype=np.uint8)
        return loaded[16:].reshape((-1, 28, 28))


def load_labels(data_dir: Path, labels_file) -> np.ndarray:
    """Load data from labels file.

    :param data_dir:directory contining data files
    :param labels_file: mnist labels file
    :return: numpy array of shape [length]
    """
    with data_dir.joinpath(labels_file).open() as handle:
        loaded = np.fromfile(file=handle, dtype=np.uint8)
        return loaded[8:]


def fetch_mnist(
    data_dir: str = "mnist", mnist_keys: Tuple[str, ...] = MNIST_KEYS
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Get dictionary with MNIST dataset.

    :param data_dir: directory with MNIST files
    :param mnist_keys: MNIST files names
    :return: dictionary with train and test dataset
    """
    data_path = Path(data_dir)
    verify_mnist_dir(data_dir=data_path, mnist_keys=mnist_keys)
    train_images, train_labels, test_images, test_labels = 4 * [np.empty(0)]
    for key in mnist_keys:
        if "train-images" in key:
            train_images = load_images(data_dir=data_path, images_file=key)
        elif "train-labels" in key:
            train_labels = load_labels(data_dir=data_path, labels_file=key)
        elif "t10k-images" in key:
            test_images = load_images(data_dir=data_path, images_file=key)
        elif "t10k-labels" in key:
            test_labels = load_labels(data_dir=data_path, labels_file=key)
    return {"train": (train_images, train_labels), "val": (test_images, test_labels)}
