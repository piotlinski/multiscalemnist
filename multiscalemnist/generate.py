"""Generate MultiScaleMNIST dataset."""
import logging
from itertools import cycle
from typing import Dict, Iterator, Optional, Tuple

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
    return int(np.random.beta(a=1, b=1) * (max_size - min_size) + min_size)


def calculate_center_coords(
    cell_index: Tuple[int, int], cell_size: Tuple[int, int]
) -> Tuple[int, int]:
    """ Calculate cell center coordinates.

    :param cell_index: selected cell index
    :param cell_size: given cell size (height, width)
    :return: given cell center coordinates (y, x)
    """
    y = int(cell_size[0] * (cell_index[0] + 0.5))
    x = int(cell_size[1] * (cell_index[1] + 0.5))
    return y, x


def randomize_center_coords(
    cell_center: Tuple[int, int], cell_size: Tuple[int, int], position_variance: float
) -> Tuple[int, int]:
    """ Get randomized coordinates for digit center.

    :param cell_center: cell center for putting the image
    :param cell_size: given cell size (height, width)
    :param position_variance: maximum position variance
    :return: digit center coordinates
    """
    y_incr = int(cell_size[0] * position_variance / 2)
    x_incr = int(cell_size[1] * position_variance / 2)
    y = random_coordinate(cell_center[0] - y_incr, cell_center[0] + y_incr)
    x = random_coordinate(cell_center[1] - x_incr, cell_center[1] + x_incr)
    return y, x


def calculate_box_coords(
    digit: np.ndarray, center_coords: Tuple[int, int], image_size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """ Calculate bounding box coordinates (x0, y0, x1, y1).

    :param digit: single digit transformed image
    :param center_coords: coordinates to put digit center at
    :param image_size: image size used to clip coordinates
    :return: bounding box coordinates: central point, width, height
    """
    y = center_coords[0] - digit.shape[0] // 2
    x = center_coords[1] - digit.shape[1] // 2
    white_ys, white_xs = np.where(digit > 0)
    return (
        max(x + white_xs.min(), 0),
        max(y + white_ys.min(), 0),
        min(x + white_xs.max(), image_size[1] - 1),
        min(y + white_ys.max(), image_size[0] - 1),
    )


def put_digit(
    image: np.ndarray, digit: np.ndarray, center_coords: Tuple[int, int]
) -> np.ndarray:
    """ Put given digit on the image at given coordinates.

    :param image: image to put digit on
    :param digit: transformed digit
    :param center_coords: coordinates where digit should be put
    :return: image with digit put on it
    """
    result = image.copy()
    y_min = center_coords[0] - digit.shape[0] // 2
    y_max = y_min + digit.shape[0]
    x_min = center_coords[1] - digit.shape[1] // 2
    x_max = x_min + digit.shape[1]
    digit_y_min = -min(0, y_min)
    digit_y_max = digit.shape[0] - max(0, y_max - image.shape[0])
    digit_x_min = -min(0, x_min)
    digit_x_max = digit.shape[1] - max(0, x_max - image.shape[1])
    image_y_min = max(0, y_min)
    image_y_max = min(image.shape[0], y_max)
    image_x_min = max(0, x_min)
    image_x_max = min(image.shape[1], x_max)
    result[image_y_min:image_y_max, image_x_min:image_x_max] += digit[
        digit_y_min:digit_y_max, digit_x_min:digit_x_max
    ]
    return np.clip(result, 0, 255)


def round_margin(margin: float, threshold: float) -> int:
    """ Round margin according to threshold

    :param margin: calculated margin
    :param threshold: threshold for rounding
    :return: rounded margin
    """
    root = int(margin)
    if threshold < margin - root:
        return root + 1
    else:
        return root


def box_to_grid_ranges(
    bounding_box: Tuple[int, int, int, int],
    grid_size: Tuple[int, int],
    image_size: Tuple[int, int],
    threshold: float,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """ Get grid index ranges filled by bounding box.

    :param bounding_box: digit bounding box (x0, y0, x1, y1)
    :param grid_size: tuple defining grid for digits
    :param image_size: output image size
    :param threshold: minimum part of cell obscured to mark as filled
    :return: tuple of ranges (min inclusive, max exclusive), obscured by bounding box
    """
    x0, y0, x1, y1 = bounding_box
    y_min = round_margin(grid_size[0] * y0 / image_size[0], threshold=1 - threshold)
    y_max = round_margin(grid_size[0] * y1 / image_size[0], threshold=threshold)
    x_min = round_margin(grid_size[1] * x0 / image_size[1], threshold=1 - threshold)
    x_max = round_margin(grid_size[1] * x1 / image_size[1], threshold=threshold)
    return (y_min, y_max), (x_min, x_max)


def mark_as_filled(
    grid: np.ndarray,
    image_size: Tuple[int, int],
    bounding_box: Tuple[int, int, int, int],
    threshold: float,
) -> np.ndarray:
    """ Mark grid cells as filled.

    :param grid: given grid array
    :param image_size: output image size
    :param bounding_box: inserted digit bounding box (x0, y0, x1, y1)
    :param threshold: minimum part of cell obscured to mark as filled
    :return:
    """
    result = grid.copy()
    (y_min, y_max), (x_min, x_max) = box_to_grid_ranges(
        bounding_box=bounding_box,
        grid_size=grid.shape,
        image_size=image_size,
        threshold=threshold,
    )
    result[y_min:y_max, x_min:x_max] = 1
    return result


def generate_image_with_annotation(
    digits: Iterator[np.ndarray],
    digit_labels: Iterator[np.ndarray],
    grid_sizes: Tuple[Tuple[int, int], ...],
    image_size: Tuple[int, int],
    n_channels: int,
    min_digit_size: int,
    position_variance: float,
    cell_filled_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Generate image with digits put in a grid of given size.

    :param digits: numpy array iterable with digits to put on the image
    :param digit_labels: numpy array iterable with digit labels
    :param grid_sizes: tuple defining available grids for digits
    :param image_size: output image size (height, width)
    :param n_channels: number of output channels
    :param min_digit_size: min size of a digit
    :param position_variance: how much digit position may vary from cell center
    :param cell_filled_threshold: minimum proportion to mark grid cell as filled
    :return: tuple: image, bounding boxes and labels
    """
    max_digits = max([np.prod(grid) for grid in grid_sizes])
    grid_size_idx = np.random.choice(range(len(grid_sizes)))
    grid_size = grid_sizes[grid_size_idx]
    grid = np.zeros(grid_size)
    image = np.zeros(image_size)
    bounding_boxes = np.full((max_digits, 4), -1)
    labels = np.full(max_digits, -1)
    cell_size = image_size[0] // grid_size[0], image_size[1] // grid_size[1]
    n_digits = np.random.randint(np.prod(grid_size) // 2, np.prod(grid_size) + 1)
    for idx in range(n_digits):
        cell_idx = random_cell(grid)
        if cell_idx is None:
            break
        digit_size = random_digit_size(
            grid=grid,
            cell_index=cell_idx,
            cell_size=cell_size,
            min_size=min_digit_size,
        )
        cell_center = calculate_center_coords(cell_index=cell_idx, cell_size=cell_size)
        digit_center_coords = randomize_center_coords(
            cell_center=cell_center,
            cell_size=cell_size,
            position_variance=position_variance,
        )
        digit = cv2.resize(
            next(digits), dsize=(digit_size, digit_size), interpolation=cv2.INTER_CUBIC,
        )
        image = put_digit(image=image, digit=digit, center_coords=digit_center_coords)
        label = next(digit_labels)
        bounding_box = calculate_box_coords(
            digit=digit, center_coords=digit_center_coords, image_size=image_size
        )
        grid = mark_as_filled(
            grid=grid,
            image_size=image_size,
            bounding_box=bounding_box,
            threshold=cell_filled_threshold,
        )
        labels[idx] = label
        bounding_boxes[idx] = bounding_box
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = np.dstack(n_channels * (image,))
    return image, bounding_boxes, labels


def generate_set(config: CfgNode, data: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """Generate entire dataset of MultiScaleMNIST."""
    max_digits = max([np.prod(grid) for grid in config.GRID_SIZES])
    with h5py.File(config.FILE_NAME, mode="w") as f:
        dataset_sizes = {"train": config.TRAIN_LENGTH, "test": config.TEST_LENGTH}
        for dataset in ["train", "test"]:
            digits, digit_labels = data[dataset]
            indices = np.random.permutation(len(digit_labels))
            digits_iter = cycle(digits[indices])
            digit_labels_iter = cycle(digit_labels[indices])
            logger.info(
                "Creating %s dataset in file %s with %d entries",
                dataset,
                config.FILE_NAME,
                dataset_sizes[dataset],
            )
            h5set = f.create_group(dataset)
            images_set = h5set.create_dataset(
                "images",
                shape=(dataset_sizes[dataset], *config.IMAGE_SIZE, config.N_CHANNELS),
                chunks=(config.CHUNK_SIZE, *config.IMAGE_SIZE, config.N_CHANNELS),
                dtype=np.uint8,
            )
            boxes_set = h5set.create_dataset(
                "boxes",
                shape=(dataset_sizes[dataset], max_digits, 4),
                chunks=(config.CHUNK_SIZE, max_digits, 4),
                dtype=np.int,
            )
            labels_set = h5set.create_dataset(
                "labels",
                shape=(dataset_sizes[dataset], max_digits),
                chunks=(config.CHUNK_SIZE, max_digits),
                dtype=np.int,
            )
            for idx in trange(dataset_sizes[dataset]):
                image, boxes, labels = generate_image_with_annotation(
                    digits=digits_iter,
                    digit_labels=digit_labels_iter,
                    grid_sizes=config.GRID_SIZES,
                    image_size=config.IMAGE_SIZE,
                    n_channels=config.N_CHANNELS,
                    min_digit_size=config.MIN_DIGIT_SIZE,
                    position_variance=config.POSITION_VARIANCE,
                    cell_filled_threshold=config.CELL_FILLED_THRESHOLD,
                )
                images_set[idx] = image
                boxes_set[idx] = boxes
                labels_set[idx] = labels
    logger.info("Done!")
