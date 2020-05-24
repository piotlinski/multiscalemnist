"""Generate MultiScaleMNIST dataset."""
import logging
from typing import Dict, Tuple

import cv2
import h5py
import numpy as np
from tqdm.auto import trange
from yacs.config import CfgNode

logger = logging.getLogger(__name__)


def random_coordinate(min_idx: int, max_idx: int):
    """Sample coordinate in a given range."""
    return np.random.randint(min_idx, max_idx)


def generate_image_with_annotation(
    config: CfgNode, digits: np.ndarray, digit_labels: np.ndarray
):
    """Generate single image with annotations."""
    n_digits = np.random.randint(config.MIN_DIGTS, config.MAX_DIGITS + 1)
    indices = np.random.choice(np.arange(len(digit_labels)), n_digits, replace=False)
    image = np.zeros(config.IMAGE_SIZE, dtype=np.uint8)
    scales = np.random.choice(config.DIGIT_SCALES, n_digits, replace=True)
    boxes = np.full((config.MAX_DIGITS, 4), -1)
    labels = np.full(config.MAX_DIGITS, -1)
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
        labels[box_idx] = digit_labels[idx]
    image = np.clip(image, 0, 255)
    return image, boxes, labels


def generate_set(config: CfgNode, data: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """Generate entire dataset of MultiScaleMNIST."""
    with h5py.File(config.FILE_NAME, mode="w") as f:
        dataset_sizes = {"train": config.TRAIN_LENGTH, "test": config.TEST_LENGTH}
        for dataset in ["train", "test"]:
            digits, digit_labels = data[dataset]
            logger.info(
                "Creating %s dataset in file %s with %d entries",
                dataset,
                config.FILE_NAME,
                dataset_sizes[dataset],
            )
            h5set = f.create_group(dataset)
            images_set = h5set.create_dataset(
                "images",
                shape=(dataset_sizes[dataset], *config.IMAGE_SIZE),
                chunks=(config.CHUNK_SIZE, *config.IMAGE_SIZE),
                dtype=np.uint8,
            )
            boxes_set = h5set.create_dataset(
                "boxes",
                shape=(dataset_sizes[dataset], config.MAX_DIGITS, 4),
                chunks=(config.CHUNK_SIZE, config.MAX_DIGITS, 4),
                dtype=np.uint8,
            )
            labels_set = h5set.create_dataset(
                "labels",
                shape=(dataset_sizes[dataset], config.MAX_DIGITS),
                chunks=(config.CHUNK_SIZE, config.MAX_DIGITS),
                dtype=np.uint8,
            )
            for idx in trange(dataset_sizes[dataset]):
                image, boxes, labels = generate_image_with_annotation(
                    config=config, digits=digits, digit_labels=digit_labels
                )
                images_set[idx] = image
                boxes_set[idx] = boxes
                labels_set[idx] = labels
    logger.info("Done!")
