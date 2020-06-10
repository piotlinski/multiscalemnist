"""Generate MultiScaleMNIST dataset."""
import logging
from typing import Dict, Optional, Tuple

import cv2
import h5py
import numpy as np
from tqdm.auto import trange
from yacs.config import CfgNode

logger = logging.getLogger(__name__)


def random_coordinate(min_idx: int, max_idx: int):
    """Sample coordinate in a given range."""
    return np.random.randint(min_idx, max_idx)


def random_cell(grid: np.ndarray) -> Optional[Tuple[int, int]]:
    """ Get random unused cell index from grid.

    :param grid: array with zeros (empty cells) and ones (full cells)
    :return: random empty cell index or None if no available
    """
    unfilled_inds = np.argwhere(grid == 0)
    if unfilled_inds.size == 0:
        return None
    idx = np.random.randint(0, unfilled_inds.shape[0])
    return unfilled_inds[idx]


def filled_margin(grid: np.ndarray, cell_index: Tuple[int, int]) -> Tuple[int, int]:
    """ Get margin from nearest filled grid cell.

    :param grid: array with zeros (empty cells) and ones (full cells)
    :param cell_index: selected cell index to put digit in
    :return: tuple of margins: (y, x)
    """
    filled_inds = np.argwhere(grid == 1)
    y_filled_margin = min(
        [
            abs(cell_index[0] - filled_idx[0])
            for filled_idx in filled_inds
            if filled_idx[1] == cell_index[1]
        ],
        default=grid.shape[0],
    )
    x_filled_margin = min(
        [
            abs(cell_index[1] - filled_idx[1])
            for filled_idx in filled_inds
            if filled_idx[0] == cell_index[0]
        ],
        default=grid.shape[1],
    )
    return y_filled_margin, x_filled_margin


def image_margin(grid: np.ndarray, cell_index: Tuple[int, int]) -> Tuple[int, int]:
    """ Get margin from grid border.

    :param grid: array with zeros (empty cells) and ones (full cells)
    :param cell_index: selected cell index to put digit in
    :return: tuple of margins: (y, x)
    """
    y_border_margin = min(cell_index[0] + 1, grid.shape[0] - cell_index[0])
    x_border_margin = min(cell_index[1] + 1, grid.shape[1] - cell_index[1])
    return y_border_margin, x_border_margin


def random_digit_size(
    grid: np.ndarray,
    cell_index: Tuple[int, int],
    cell_size: Tuple[int, int],
    min_size: int,
) -> int:
    """ Get random digit size that will fit the given cell and its surroundings.

    :param grid: array with zeros (empty cells) and ones (full cells)
    :param cell_index: selected cell index to put digit in
    :param cell_size: given cell size (height, width)
    :param min_size: minimal size of returned digit
    :return: random digit size (pixels) that will fit in given place
    """
    y_image_margin, x_image_margin = image_margin(grid=grid, cell_index=cell_index)
    y_filled_margin, x_filled_margin = filled_margin(grid=grid, cell_index=cell_index)
    margin = (
        min(y_image_margin, y_filled_margin),
        min(x_image_margin, x_filled_margin),
    )
    max_size = min(
        int(cell_size[0] * (2 * margin[0] - 1)), int(cell_size[1] * (2 * margin[1] - 1))
    )
    if max_size < min_size:
        return min_size
    return np.random.randint(min_size, max_size)


def calculate_center_coords(
    cell_index: Tuple[int, int], grid_size: Tuple[int, int], image_size: Tuple[int, int]
) -> Tuple[int, int]:
    """ Calculate cell center coordinates.

    :param cell_index: selected cell index
    :param grid_size: grid size (height, width)
    :param image_size: image size (height, width)
    :return: given cell center coordinates (y, x)
    """


def randomize_center_coords(
    cell_center: Tuple[int, int], cell_size: Tuple[int, int], position_variance: float
) -> Tuple[int, int]:
    """ Get randomized coordinates for digit center.

    :param cell_center: cell center for putting the image
    :param cell_size: given cell size (height, width)
    :param position_variance: maximum position variance
    :return: digit center coordinates
    """


def calculate_box_coords(
    digit: np.ndarray, center_coords: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """ Calculate bounding box coordinates (x, y, w, h).

    :param digit: single digit transformed image
    :param center_coords: coordinates to put digit center at
    :return: bounding box coordinates: central point, width, height
    """


def put_digit(
    image: np.ndarray, digit: np.ndarray, center_coords: Tuple[int, int]
) -> np.ndarray:
    """ Put given digit on the image at given coordinates.

    :param image: image to put digit on
    :param digit: transformed digit
    :param center_coords: coordinates where digit should be put
    :return: image with digit put on it
    """


def mark_as_filled(
    grid: np.ndarray,
    image_size: Tuple[int, int],
    bounding_box: Tuple[int, int, int, int],
    threshold: float,
) -> np.ndarray:
    """ Mark grid cells as filled.

    :param grid: given grid array
    :param image_size: output image size
    :param bounding_box: inserted digit bounding box (x, y, w, h)
    :param threshold: minimum part of cell obscured to mark as filled
    :return:
    """


def generate_image_with_annotation(
    digits: np.ndarray,
    digit_labels: np.ndarray,
    grid_size: Tuple[int, int],
    image_size: Tuple[int, int],
    position_variance: float,
    cell_filled_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Generate image with digits put in a grid of given size.

    :param digits: numpy array with digits to put on the image
    :param digit_labels: numpy array with digit labels
    :param grid_size: tuple defining grid for digits
    :param image_size: output image size (height, width)
    :param position_variance: how much digit position may vary from cell center
    :param cell_filled_threshold: minimum proportion to mark grid cell as filled
    :return: tuple: image, bounding boxes and labels
    """
    grid = np.zeros(grid_size)
    image = np.zeros(image_size)
    bounding_boxes = np.full((np.prod(grid_size), 4), -1)
    labels = np.full(np.prod(grid_size), -1)
    cell_size = image_size[0] // grid_size[0], image_size[1] // grid_size[1]
    n_digits = np.random.randint(np.prod(grid_size) // 2, np.prod(grid_size) + 1)
    indices = np.random.choice(np.arange(len(digit_labels)), n_digits, replace=False)
    for idx, digit_idx in enumerate(indices):
        cell_idx = random_cell(grid)
        if cell_idx is None:
            break
        digit_size = random_digit_size(
            grid=grid,
            cell_index=cell_idx,
            cell_size=cell_size,
            min_size=digits.shape[-1:],
        )
        cell_center = calculate_center_coords(
            cell_index=cell_idx, grid_size=grid_size, image_size=image_size
        )
        digit_center_coords = randomize_center_coords(
            cell_center=cell_center,
            cell_size=cell_size,
            position_variance=position_variance,
        )
        digit = cv2.resize(
            digits[digit_idx],
            dsize=(digit_size, digit_size),
            interpolation=cv2.INTER_CUBIC,
        )
        image = put_digit(image=image, digit=digit, center_coords=digit_center_coords)
        label = digit_labels[digit_idx]
        bounding_box = calculate_box_coords(
            digit=digit, center_coords=digit_center_coords
        )
        grid = mark_as_filled(
            grid=grid,
            image_size=image_size,
            bounding_box=bounding_box,
            threshold=cell_filled_threshold,
        )
        labels[idx] = label
        bounding_boxes[idx] = bounding_box
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8), bounding_boxes, labels


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
                shape=(dataset_sizes[dataset], np.prod(config.GRID_SIZE), 4),
                chunks=(config.CHUNK_SIZE, np.prod(config.GRID_SIZE), 4),
                dtype=np.int,
            )
            labels_set = h5set.create_dataset(
                "labels",
                shape=(dataset_sizes[dataset], np.prod(config.GRID_SIZE)),
                chunks=(config.CHUNK_SIZE, np.prod(config.GRID_SIZE)),
                dtype=np.int,
            )
            for idx in trange(dataset_sizes[dataset]):
                image, boxes, labels = generate_image_with_annotation(
                    digits=digits,
                    digit_labels=digit_labels,
                    grid_size=config.GRID_SIZE,
                    image_size=config.IMAGE_SIZE,
                    position_variance=config.POSITION_VARIANCE,
                    cell_filled_threshold=config.CELL_FILLED_THRESHOLD,
                )
                images_set[idx] = image
                boxes_set[idx] = boxes
                labels_set[idx] = labels
    logger.info("Done!")
