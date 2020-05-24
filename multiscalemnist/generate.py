"""Generate MultiScaleMNIST dataset."""
import logging

import cv2
import numpy as np
from yacs.config import CfgNode

logger = logging.getLogger(__name__)


def random_coordinate(min_idx: int, max_idx: int):
    """Sample coordinate in a given range."""
    return np.random.randint(min_idx, max_idx)


def generate_image_with_annotation(
    config: CfgNode, digits: np.ndarray, labels: np.ndarray
):
    """Generate single image with annotations."""
    n_digits = np.random.randint(config.MIN_DIGTS, config.MAX_DIGITS + 1)
    indices = np.random.choice(np.arange(len(labels)), n_digits, replace=False)
    image = np.zeros(config.IMAGE_SIZE, dtype=np.uint8)
    scales = np.random.choice(config.DIGIT_SCALES, n_digits, replace=True)
    boxes = np.empty((len(indices), 4))
    for box_idx, (idx, scale) in enumerate(zip(indices, scales)):
        x_size, y_size = config.DIGIT_SIZE[0] * scale, config.DIGIT_SIZE[1] * scale
        y_coord = random_coordinate(0, config.IMAGE_SIZE[0] - y_size)
        x_coord = random_coordinate(0, config.IMAGE_SIZE[1] - x_size)
        digit = cv2.resize(
            digits[idx], dsize=(x_size, y_size), interpolation=cv2.INTER_CUBIC,
        )
        white_ys, white_xs = np.where(digit > 0)
        image[y_coord : y_coord + y_size, x_coord : x_coord + x_size] += digit
        boxes[box_idx] = [
            x_coord + white_xs.min(),
            y_coord + white_ys.min(),
            x_coord + white_xs.max(),
            y_coord + white_ys.max(),
        ]
    image = np.clip(image, 0, 255)
    return image, boxes, labels[indices]
