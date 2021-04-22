"""Test MultiScaleMNIST generating."""
from unittest.mock import patch

import numpy as np
import pytest

from multiscalemnist.generate import (
    box_to_grid_ranges,
    calculate_box_coords,
    calculate_center_coords,
    filled_margin,
    filter_digits,
    generate_image_with_annotation,
    generate_set,
    image_margin,
    mark_as_filled,
    put_digit,
    random_cell,
    random_coordinate,
    random_digit_size,
    randomize_center_coords,
    round_margin,
)


@pytest.mark.parametrize("min_val, max_val", [(0, 3), (5, 12), (120, 332)])
def test_random_coordinate(min_val, max_val):
    """Test drawing random coordinate."""
    assert min_val <= random_coordinate(min_val, max_val) < max_val


@pytest.mark.parametrize("_iter", range(5))
@pytest.mark.parametrize("grid_size", [[2, 2], [5, 5], [3, 7]])
def test_random_cell(grid_size, _iter):
    """Test drawing random indices in grid."""
    grid = np.zeros(grid_size)
    y_idx, x_idx = random_cell(grid)
    assert 0 <= y_idx < grid_size[0]
    assert 0 <= x_idx < grid_size[1]


@pytest.mark.parametrize("_iter", range(5))
def test_draw_indices_with_ones(_iter):
    """Test if nonzero cell is not drawn."""
    grid = np.zeros((3, 3))
    nonzero_inds = random_cell(grid)
    grid[nonzero_inds] = 1
    inds = random_cell(grid)
    assert not (inds[0] == nonzero_inds[0] and inds[1] == nonzero_inds[1])


def test_draw_indices_ones():
    """Test if None returned when no indices can be drawn."""
    grid = np.ones((3, 3))
    assert random_cell(grid) is None


@pytest.mark.parametrize(
    "nonzero, cell_idx, expected",
    [
        ([], (2, 2), (5, 5)),
        ([(1, 1)], (2, 1), (1, 5)),
        ([(1, 1)], (1, 2), (5, 1)),
        ([(2, 0), (1, 2)], (2, 2), (1, 2)),
    ],
)
def test_filled_margin(nonzero, cell_idx, expected):
    """Test filled margin."""
    grid = np.zeros((5, 5))
    for y_idx, x_idx in nonzero:
        grid[y_idx, x_idx] = 1
    assert filled_margin(grid, cell_idx) == expected


@pytest.mark.parametrize(
    "grid_size, cell_idx, expected",
    [
        ((3, 3), (1, 1), (2, 2)),
        ((4, 5), (2, 1), (2, 2)),
        ((9, 9), (2, 8), (3, 1)),
        ((11, 11), (7, 0), (4, 1)),
    ],
)
def test_image_margin(grid_size, cell_idx, expected):
    """Test image margin."""
    grid = np.zeros(grid_size)
    assert image_margin(grid, cell_idx) == expected


@pytest.mark.parametrize("_iter", range(5))
def test_random_digit_size(_iter):
    """Verify drawing random digit size."""
    grid = np.zeros((3, 3))
    cell_index = (1, 1)
    cell_size = (20, 20)
    min_size = 5
    max_size = 10000
    digit_size = random_digit_size(
        grid=grid,
        cell_index=cell_index,
        cell_size=cell_size,
        min_size=min_size,
        max_size=max_size,
    )
    assert min_size <= digit_size < 60


def test_random_digit_size_too_small():
    """Verify if min size returned when too small."""
    grid = np.zeros((3, 3))
    cell_index = (0, 0)
    cell_size = (20, 20)
    min_size = 28
    max_size = 10000
    digit_size = random_digit_size(
        grid=grid,
        cell_index=cell_index,
        cell_size=cell_size,
        min_size=min_size,
        max_size=max_size,
    )
    assert digit_size == min_size


def test_random_digit_size_too_big():
    """Verify if max size returned when too big."""
    grid = np.zeros((3, 3))
    cell_index = (0, 0)
    cell_size = (20, 20)
    min_size = 1
    max_size = 1
    digit_size = random_digit_size(
        grid=grid,
        cell_index=cell_index,
        cell_size=cell_size,
        min_size=min_size,
        max_size=max_size,
    )
    assert digit_size == min_size


@pytest.mark.parametrize(
    "cell_index, cell_size, expected",
    [
        ((0, 0), (20, 20), (10, 10)),
        ((3, 2), (25, 10), (87, 25)),
        ((10, 11), (28, 28), (294, 322)),
    ],
)
def test_calculate_center_coords(cell_index, cell_size, expected):
    """Verify cell center coords calculation."""
    assert (
        calculate_center_coords(cell_index=cell_index, cell_size=cell_size) == expected
    )


@pytest.mark.parametrize(
    "cell_center, cell_size, position_variance, y_range, x_range",
    [
        ((150, 150), (20, 20), 0.5, (145, 155), (145, 155)),
        ((127, 174), (60, 60), 0.8, (103, 151), (150, 198)),
    ],
)
def test_randomize_center_coords(
    cell_center, cell_size, position_variance, y_range, x_range
):
    """Verify if randomized center coords fit expected range."""
    y, x = randomize_center_coords(
        cell_center=cell_center,
        cell_size=cell_size,
        position_variance=position_variance,
    )
    assert y_range[0] <= y <= y_range[1]
    assert x_range[0] <= x <= x_range[1]


@pytest.mark.parametrize("x_center, y_center", [(50, 50), (120, 115), (174, 178)])
@pytest.mark.parametrize(
    "x0, y0, x1, y1", [(6, 14, 9, 15), (77, 53, 92, 65), (40, 20, 60, 40)]
)
def test_calculate_box_coords(x_center, y_center, x0, y0, x1, y1):
    """Test calculating bounding box coordinates."""
    image_size = (250, 250)
    digit = np.zeros((100, 100))
    x = x_center - 50
    y = y_center - 50
    digit[y0:y1, x0:x1] = 1
    bounding_box = calculate_box_coords(
        digit, center_coords=(y_center, x_center), image_size=image_size
    )
    assert bounding_box == (x + x0, y + y0, x + x1 - 1, y + y1 - 1)


def test_clipping_box_coords():
    """Test if incorrect bounding boxes are clipped."""
    image_size = (100, 100)
    digit = np.ones((120, 120))
    bounding_box = calculate_box_coords(
        digit, center_coords=(50, 50), image_size=image_size
    )
    assert bounding_box == (0, 0, 99, 99)


@pytest.mark.parametrize(
    "digit_size, center_coords",
    [
        ((10, 10), (8, 8)),
        ((24, 32), (50, 50)),
        ((5, 5), (12, 17)),
        ((35, 53), (29, 13)),
        ((20, 30), (90, 90)),
    ],
)
def test_put_digit(digit_size, center_coords):
    """Verify putting digit on an image."""
    image = np.zeros((100, 100))
    digit = np.ones(digit_size)
    new_image = put_digit(image=image, digit=digit, center_coords=center_coords)
    negative_y_incr = digit_size[0] // 2
    negative_x_incr = digit_size[1] // 2
    top_left_y = max(0, center_coords[0] - negative_y_incr)
    top_left_x = max(0, center_coords[1] - negative_x_incr)
    bottom_right_y = min(
        image.shape[0], center_coords[0] + digit_size[0] - negative_y_incr
    )
    bottom_right_x = min(
        image.shape[1], center_coords[1] + digit_size[1] - negative_x_incr
    )
    assert np.all(new_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] == 1)
    new_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
    assert np.all(new_image == image)


@pytest.mark.parametrize(
    "margin, threshold, expected",
    [(2.3, 0.5, 2), (4.7, 0.5, 5), (13.6, 0.6, 13), (27.3, 0.2, 28)],
)
def test_round_margin(margin, threshold, expected):
    """Test rounding mark margin."""
    assert round_margin(margin=margin, threshold=threshold) == expected


@pytest.mark.parametrize(
    "bounding_box, grid_size, image_size, threshold, expected",
    [
        ((5, 5, 15, 15), (8, 8), (80, 80), 0.8, ((1, 1), (1, 1))),
        ((1, 1, 9, 9), (4, 4), (20, 20), 0.1, ((0, 2), (0, 2))),
        ((90, 90, 100, 100), (5, 5), (100, 100), 0.5, ((4, 5), (4, 5))),
        ((24, 11, 50, 45), (7, 7), (70, 70), 0.5, ((1, 4), (2, 5))),
        ((24, 11, 50, 45), (7, 7), (70, 70), 0.3, ((1, 5), (2, 5))),
    ],
)
def test_box_to_grid_ranges(bounding_box, image_size, grid_size, threshold, expected):
    """Test getting obscured grid ranges from bounding box."""
    ranges = box_to_grid_ranges(
        bounding_box=bounding_box,
        grid_size=grid_size,
        image_size=image_size,
        threshold=threshold,
    )
    assert ranges == expected


@pytest.mark.parametrize(
    "grid_size, image_size, bounding_box, threshold, filled_ranges",
    [
        ((8, 8), (80, 80), (5, 5, 15, 15), 0.0, ((0, 2), (0, 2))),
        ((4, 4), (20, 20), (1, 1, 9, 9), 0.1, ((0, 2), (0, 2))),
        ((5, 5), (100, 100), (90, 90, 100, 100), 0.5, ((4, 5), (4, 5))),
        ((7, 7), (70, 70), (24, 11, 50, 45), 0.5, ((1, 4), (2, 5))),
        ((7, 7), (70, 70), (24, 11, 50, 45), 0.3, ((1, 5), (2, 5))),
    ],
)
def test_mark_as_filled(grid_size, image_size, bounding_box, threshold, filled_ranges):
    """Verify marking grid as filled."""
    grid = np.zeros(grid_size)
    filled_grid = mark_as_filled(grid, image_size, bounding_box, threshold)
    ((y_min, y_max), (x_min, x_max)) = filled_ranges
    assert np.all(filled_grid[y_min:y_max, x_min:x_max] == 1)
    filled_grid[y_min:y_max, x_min:x_max] = 0
    assert np.all(filled_grid == grid)


@pytest.mark.parametrize(
    "grid_sizes, image_size, n_channels",
    [
        (((3, 3), (2, 2)), (100, 100), 1),
        (((8, 8),), (64, 64), 3),
        (((5, 8), (2, 4)), (40, 64), 5),
    ],
)
def test_generate_image_with_annotation(grid_sizes, image_size, n_channels):
    """Test generating image with annotation."""
    n_digits = max([np.prod(grid_size) for grid_size in grid_sizes])
    digits = iter(np.random.randint(0, 256, (n_digits, 28, 28), dtype=np.uint8))
    digit_labels = iter(np.random.randint(0, 10, (n_digits,), dtype=np.uint8))
    image, boxes, labels = generate_image_with_annotation(
        digits=digits,
        digit_labels=digit_labels,
        grid_sizes=grid_sizes,
        image_size=image_size,
        n_channels=n_channels,
        min_digit_size=32,
        max_digit_size=10000,
        position_variance=0.5,
        cell_filled_threshold=0.5,
    )
    assert image.shape == (*image_size, n_channels)
    assert boxes.shape[0] == labels.shape[0]
    assert boxes.shape[1] == 4


@pytest.mark.parametrize("digit_set", [(1, 2), (1, 3), (2, 4)])
def test_filter_digits(digit_set):
    """Verify if filtered digits are dropped."""
    digits = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    labels = np.array([2, 2, 4, 3, 1, 1, 3, 2, 1])
    filtered_digits, filtered_labels = filter_digits(digits, labels, digit_set)
    assert len(filtered_digits) == len(filtered_labels) <= len(digits)
    assert all([v in digit_set for v in filtered_labels])
    discarded = set(labels) - set(digit_set)
    assert all([v not in discarded for v in filtered_labels])


@patch("multiscalemnist.generate.cv2.imwrite")
@patch("multiscalemnist.generate.Path.mkdir")
@patch("multiscalemnist.generate.Path.open")
def test_generating_entire_dataset(
    open_mock, _mkdir_mock, _imwrite_mock, sample_config
):
    """Test generating entire dataset."""
    n_digits = max([np.prod(grid) for grid in sample_config.GRID_SIZES])
    digits = np.random.randint(0, 256, (n_digits, 28, 28), dtype=np.uint8)
    digit_labels = np.random.randint(0, 10, (n_digits,), dtype=np.uint8)
    data = {"train": (digits, digit_labels), "val": (digits, digit_labels)}
    generate_set(sample_config, data)
    enter_mock = open_mock.return_value.__enter__
    enter_mock.assert_called()
