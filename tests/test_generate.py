"""Test MultiScaleMNIST generating."""
from unittest.mock import call, patch

import numpy as np
import pytest

from multiscalemnist.generate import (
    box_to_grid_ranges,
    calculate_box_coords,
    calculate_center_coords,
    filled_margin,
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
    digit_size = random_digit_size(
        grid=grid, cell_index=cell_index, cell_size=cell_size, min_size=min_size,
    )
    assert min_size <= digit_size < 60


def test_random_digit_size_too_small():
    """Verify if min size returned when too small."""
    grid = np.zeros((3, 3))
    cell_index = (0, 0)
    cell_size = (20, 20)
    min_size = 28
    digit_size = random_digit_size(
        grid=grid, cell_index=cell_index, cell_size=cell_size, min_size=min_size,
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


@pytest.mark.parametrize(
    "x, y, w, h", [(10, 12, 8, 6), (125, 43, 28, 28), (433, 674, 128, 128)]
)
def test_calculate_box_coords(x, y, w, h):
    """Test calculating bounding box coordinates."""
    image = np.zeros((h, w))
    center_coords = (y, x)
    assert calculate_box_coords(image, center_coords=center_coords) == (x, y, w, h)


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
        ((10, 10, 10, 10), (8, 8), (80, 80), 0.8, ((1, 1), (1, 1))),
        ((5, 5, 8, 8), (4, 4), (20, 20), 0.1, ((0, 2), (0, 2))),
        ((95, 95, 10, 10), (5, 5), (100, 100), 0.5, ((4, 5), (4, 5))),
        ((37, 28, 26, 34), (7, 7), (70, 70), 0.5, ((1, 4), (2, 5))),
        ((37, 28, 26, 34), (7, 7), (70, 70), 0.3, ((1, 5), (2, 5))),
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
        ((8, 8), (80, 80), (10, 10, 10, 10), 0.0, ((0, 2), (0, 2))),
        ((4, 4), (20, 20), (5, 5, 8, 8), 0.1, ((0, 2), (0, 2))),
        ((5, 5), (100, 100), (95, 95, 10, 10), 0.5, ((4, 5), (4, 5))),
        ((7, 7), (70, 70), (37, 28, 26, 34), 0.5, ((1, 4), (2, 5))),
        ((7, 7), (70, 70), (37, 28, 26, 34), 0.3, ((1, 5), (2, 5))),
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
    "grid_size, image_size",
    [((3, 3), (100, 100)), ((8, 8), (64, 64)), ((5, 8), (40, 64))],
)
def test_generate_image_with_annotation(grid_size, image_size):
    """Test generating image with annotation."""
    n_digits = np.prod(grid_size)
    digits = np.random.randint(0, 256, (n_digits, 28, 28), dtype=np.uint8)
    digit_labels = np.random.randint(0, 10, (n_digits,), dtype=np.uint8)
    image, boxes, labels = generate_image_with_annotation(
        digits=digits,
        digit_labels=digit_labels,
        grid_size=grid_size,
        image_size=image_size,
        position_variance=0.5,
        cell_filled_threshold=0.5,
    )
    assert image.shape == image_size
    assert boxes.shape[0] == labels.shape[0] == n_digits
    assert boxes.shape[1] == 4


@patch("multiscalemnist.generate.h5py.File")
def test_generating_entire_dataset(h5_mock, sample_config):
    """Test generating entire dataset with H5."""
    n_digits = np.prod(sample_config.GRID_SIZE)
    digits = np.random.randint(0, 256, (n_digits, 28, 28), dtype=np.uint8)
    digit_labels = np.random.randint(0, 10, (n_digits,), dtype=np.uint8)
    data = {"train": (digits, digit_labels), "test": (digits, digit_labels)}
    generate_set(sample_config, data)
    enter_mock = h5_mock.return_value.__enter__
    enter_mock.assert_called()
    enter_mock.return_value.create_group.assert_has_calls(
        [call("train"), call("test")], any_order=True
    )
