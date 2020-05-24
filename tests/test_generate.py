"""Test MultiScaleMNIST generating."""
from unittest.mock import call, patch

import numpy as np
import pytest

from multiscalemnist.generate import (
    generate_image_with_annotation,
    generate_set,
    random_coordinate,
)


@pytest.mark.parametrize("min_val, max_val", [(0, 3), (5, 12), (120, 332)])
def test_random_coordinate(min_val, max_val):
    """Test drawing random coordinate."""
    assert min_val <= random_coordinate(min_val, max_val) < max_val


def test_generate_image_with_annotation(sample_config):
    """Test generating image with annotation."""
    digits = np.random.randint(0, 256, (3, 28, 28), dtype=np.uint8)
    digit_labels = np.random.randint(0, 10, (3,), dtype=np.uint8)
    image, boxes, labels = generate_image_with_annotation(
        sample_config, digits, digit_labels
    )
    assert image.shape == sample_config.IMAGE_SIZE
    assert boxes.shape[0] == labels.shape[0]
    assert boxes.shape[1] == 4


@patch("multiscalemnist.generate.h5py.File")
def test_generating_entire_dataset(h5_mock, sample_config):
    """Test generating entire dataset with H5."""
    digits = np.random.randint(0, 256, (3, 28, 28), dtype=np.uint8)
    digit_labels = np.random.randint(0, 10, (3,), dtype=np.uint8)
    data = {"train": (digits, digit_labels), "test": (digits, digit_labels)}
    generate_set(sample_config, data)
    enter_mock = h5_mock.return_value.__enter__
    enter_mock.assert_called()
    enter_mock.return_value.create_group.assert_has_calls(
        [call("train"), call("test")], any_order=True
    )
