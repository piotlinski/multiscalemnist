"""Test MultiScaleMNIST generating."""
import numpy as np
import pytest

from multiscalemnist.generate import generate_image_with_annotation, random_coordinate


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
