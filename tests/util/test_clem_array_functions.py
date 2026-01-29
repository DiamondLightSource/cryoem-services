from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest

from cryoemservices.util.clem_array_functions import (
    LUT,
    align_image_to_reference,
    align_image_to_self,
    convert_to_rgb,
    flatten_image,
    is_grayscale_image,
    is_image_stack,
    merge_images,
)

is_image_stack_test_matrix = (
    # Shape | Outcome
    # These should all pass
    # Grayscale
    ((1, 64, 64), True),
    ((5, 64, 64), True),
    # RGB/RGBA
    ((1, 64, 64, 3), True),
    ((5, 64, 64, 3), True),
    ((1, 64, 64, 4), True),
    ((5, 64, 64, 4), True),
    # These should all fail but not error
    # Grayscale
    ((64, 64), False),
    # RGB/RGBA
    ((64, 64, 3), False),
    ((64, 64, 4), False),
)


@pytest.mark.parametrize("test_params", is_image_stack_test_matrix)
def test_is_image_stack(test_params: tuple[tuple[int, ...], bool]):
    # Unpack test params
    shape, result = test_params

    # Create test image array
    array = np.zeros(shape)

    # Check that it's correct
    assert is_image_stack(array) is result


is_image_stack_error_cases = (
    # Shape
    # These shapes should all error
    ((64,),),
    ((5, 64, 64, 5),),
    ((5, 64, 64, 64, 5),),
)


@pytest.mark.parametrize("test_params", is_image_stack_error_cases)
def test_is_image_stack_errors(test_params: tuple[tuple[int, ...]]):
    # Unpack test params
    (shape,) = test_params

    # Create test image array
    array = np.zeros(shape)

    # Check that it errors
    with pytest.raises(ValueError):
        is_image_stack(array)


self_alignment_test_matrix = (
    # No. frames | x-offset | y-offset | Alignment starting point | dtype
    # Test uint8 arrays
    (10, 1, 1, "beginning", "uint8"),
    (10, 1, 1, "middle", "uint8"),
    (10, 1, 1, "end", "uint8"),
    (10, 2, 2, "beginning", "uint8"),
    (10, 2, 2, "middle", "uint8"),
    (10, 2, 2, "end", "uint8"),
    (10, -1, -1, "beginning", "uint8"),
    (10, -1, -1, "middle", "uint8"),
    (10, -1, -1, "end", "uint8"),
    (10, -2, -2, "beginning", "uint8"),
    (10, -2, -2, "middle", "uint8"),
    (10, -2, -2, "end", "uint8"),
    # Test float arrays
    (10, 1, 1, "beginning", "float64"),
    (10, 1, 1, "middle", "float64"),
    (10, 1, 1, "end", "float64"),
    (10, 2, 2, "beginning", "float64"),
    (10, 2, 2, "middle", "float64"),
    (10, 2, 2, "end", "float64"),
    (10, -2, 2, "beginning", "float64"),
    (10, -2, 2, "middle", "float64"),
    (10, -2, 2, "end", "float64"),
    (10, 1, -2, "beginning", "float64"),
    (10, 1, -2, "middle", "float64"),
    (10, 1, -2, "end", "float64"),
)


@pytest.mark.parametrize("test_params", self_alignment_test_matrix)
def test_align_image_to_self(
    test_params: tuple[int, int, int, Literal["beginning", "middle", "end"], str],
):
    # Unpack test params
    frames, x_offset, y_offset, start_point, dtype = test_params

    # Create a reference image stack with offset bright spots
    array = np.zeros((frames, 64, 64), dtype=dtype)

    # Add 3(?) bright spots per frame at different offsets
    for f in range(frames):
        array[f][24 + (y_offset * f)][24 + (x_offset * f)] = 255
        array[f][32 + (y_offset * f)][36 + (x_offset * f)] = 255
        array[f][36 + (y_offset * f)][24 + (x_offset * f)] = 255

    # Align the frames in the stack
    aligned = align_image_to_self(array=array, start_from=start_point)

    # Assert that bright spots are aligned throughout the stack
    for f in range(frames):
        if f == 0:
            continue
        np.testing.assert_allclose(
            aligned[f - 1],
            aligned[f],
            rtol=1,
            atol=5,
            # Absolute intensities can change by quite a bit during alignment
        )


cross_alignment_test_matrix = (
    # x-offset | y-offset | dtype
    (1, 1, "uint8"),
    (2, 2, "uint8"),
    (-1, 1, "uint8"),
    (-2, 2, "uint8"),
    (-1, 1, "float64"),
    (-2, 2, "float64"),
    (-1, -1, "float64"),
    (-2, -2, "float64"),
)


@pytest.mark.parametrize("test_params", cross_alignment_test_matrix)
def test_align_image_to_reference(test_params: tuple[int, int, str]):
    def gaussian_2d(
        shape: tuple[int, int],
        amplitude: float,
        centre: tuple[int, int],
        sigma: tuple[float, float],
        theta: float,
        offset: float,
    ):
        x0, y0 = centre
        sig_x, sig_y = sigma

        # Create meshgrid
        rows, cols = shape
        y, x = np.meshgrid(np.arange(cols), np.arange(rows), indexing="ij")

        x_rot: np.ndarray = (x - x0) * np.cos(np.deg2rad(theta)) + (y - y0) * np.sin(
            np.deg2rad(theta)
        )
        y_rot: np.ndarray = (y - y0) * np.cos(np.deg2rad(theta)) - (x - x0) * np.sin(
            np.deg2rad(theta)
        )

        # Compute and return Gaussian
        gaussian = (
            amplitude
            * np.exp(-(x_rot**2 / (2 * sig_x**2)) - (y_rot**2 / (2 * sig_y**2)))
            + offset
        )

        return gaussian

    # Unpack test params
    x_offset, y_offset, dtype = test_params

    n_frames = 5
    shape = (64, 64)
    ref = np.zeros((n_frames, *shape), dtype=dtype)
    mov = np.zeros((n_frames, *shape), dtype=dtype)

    for f in range(n_frames):
        # Add bright spots
        # ref[f][24][24] = 255
        # ref[f][36][48] = 255
        # ref[f][48][36] = 255

        # Add bright spots with offset
        # mov[f][24 + y_offset][24 + x_offset] = 255
        # mov[f][36 + y_offset][48 + x_offset] = 255
        # mov[f][48 + y_offset][36 + x_offset] = 255

        # Add 2(?) Gaussian peaks to arrays
        ref[f] += (
            gaussian_2d(
                shape,
                200,
                (24, 24),
                (2, 4),
                30,
                0,
            )
            + gaussian_2d(
                shape,
                150,
                (48, 48),
                (6, 2),
                45,
                0,
            )
        ).astype(dtype)

        mov[f] += (
            gaussian_2d(
                shape,
                200,
                (24 + x_offset, 24 + y_offset),
                (2, 4),
                30,
                0,
            )
            + gaussian_2d(
                shape,
                150,
                (48 + x_offset, 48 + y_offset),
                (6, 2),
                45,
                0,
            )
        ).astype(dtype)

    # Align moving image to reference
    reg = align_image_to_reference(ref, mov)

    # Assert that bright spots are aligned
    np.testing.assert_allclose(
        ref,
        reg,
        rtol=1,
        atol=5,
        # Absolute intensities can change by quite a bit during alignment
    )


is_grayscale_image_test_matrix = (
    # Shape | Result
    # These should pass
    ((64, 64), True),
    ((1, 64, 64), True),
    ((5, 64, 64), True),
    # These should fail without erroring
    ((64, 64, 3), False),
    ((64, 64, 4), False),
    ((1, 64, 64, 3), False),
    ((5, 64, 64, 3), False),
    ((1, 64, 64, 4), False),
    ((5, 64, 64, 4), False),
)


@pytest.mark.parametrize("test_params", is_grayscale_image_test_matrix)
def test_is_grayscale_image(test_params: tuple[tuple[int, ...], bool]):
    # Unpack test params
    shape, result = test_params

    # Construct array
    array = np.zeros(shape)

    # Check that the result is as expected
    assert is_grayscale_image(array) is result


is_grayscale_image_error_cases = (
    # Shape
    # These should all error
    ((64,),),
    ((5, 5, 64, 64),),
    ((5, 64, 64, 5),),
    ((5, 5, 5, 64, 64),),
    ((5, 5, 64, 64, 5),),
)


@pytest.mark.parametrize("test_params", is_grayscale_image_error_cases)
def test_is_grayscale_image_errors(test_params: tuple[tuple[int, ...]]):
    # Unpack test params
    (shape,) = test_params

    # Create array
    array = np.zeros(shape)

    with pytest.raises(ValueError):
        is_grayscale_image(array)


image_coloring_test_matrix = (
    # Colour | dtype | frames
    # 0
    ("red", "uint8", 1),
    ("green", "int16", 5),
    ("blue", "uint32", 1),
    ("cyan", "int64", 5),
    ("magenta", "float16", 1),
    # 5
    ("yellow", "float32", 5),
    ("gray", "float64", 1),
    ("red", "float128", 5),
    ("green", "complex64", 1),
    ("blue", "complex128", 5),
    # 10
    ("cyan", "complex256", 1),
    ("magenta", "complex64", 5),
    ("yellow", "complex128", 1),
    ("gray", "complex256", 5),
    ("red", "float16", 1),
    # 15
    ("green", "float32", 5),
    ("blue", "float64", 1),
    ("cyan", "float128", 5),
    ("magenta", "int8", 1),
    ("yellow", "uint16", 5),
    # 20
    ("gray", "int32", 1),
)


@pytest.mark.parametrize("test_params", image_coloring_test_matrix)
def test_convert_to_rgb(test_params: tuple[str, str, int]):
    def create_test_array(shape: tuple, frames: int, dtype: str):
        for f in range(frames):
            frame = np.ones(shape).astype(dtype)
            if f == 0:
                arr = np.array([frame])
            else:
                arr = np.append(arr, [frame], axis=0)
        return arr

    # Unpack parameters
    color, dtype, frames = test_params

    # Create test grayscale array
    shape = (64, 64)
    shape_new = (frames, *shape, 3)
    arr = create_test_array(shape, frames, dtype)

    # Test both success and failure conditions
    arr_new = convert_to_rgb(arr, color)

    # Check that array has expected properties
    assert arr_new.shape == shape_new
    assert str(arr_new.dtype) == dtype

    # Check that RGB values have been applied correctly
    pixel_values = np.array(LUT[color].value).astype(dtype)
    assert np.all(arr_new[0][0][0] == pixel_values)


image_coloring_fail_cases = (
    ("black", "uint8", 1),
    ("white", "int16", 5),
    ("orange", "uint32", 1),
    ("indigo", "int64", 5),
    ("violet", "float64", 1),
    ("pneumonoultramicroscopicsilicovolcanoconiosis", "complex128", 5),
)


@pytest.mark.parametrize("test_params", image_coloring_fail_cases)
def test_convert_to_rgb_fails(test_params: tuple[str, str, int]):
    def create_test_array(shape: tuple, frames: int, dtype: str) -> np.ndarray:
        for f in range(frames):
            frame = np.ones(shape).astype(dtype)
            if f == 0:
                arr = np.array([frame])
            else:
                arr = np.append(arr, [frame], axis=0)
        return arr

    # The colours should throw up an error
    with pytest.raises(KeyError):
        # Unpack parameters
        color, dtype, frames = test_params
        shape = (64, 64)
        arr = create_test_array(shape, frames, dtype)

        convert_to_rgb(arr, color)


image_flattening_test_matrix = (
    # Type | Frames | Mode | Is float? | Expected pixel value
    # Test grayscale images
    ("gray", 5, "mean", True, 2),
    ("gray", 5, "min", False, 0),
    ("gray", 5, "max", True, 4),
    ("gray", 5, None, True, 2),
    # Test RGB images
    ("rgb", 5, "mean", False, 2),
    ("rgb", 5, "min", True, 0),
    ("rgb", 5, "max", False, 4),
    ("rgb", 5, None, False, 2),
)


@pytest.mark.parametrize("test_params", image_flattening_test_matrix)
def test_flatten_image(
    test_params: tuple[str, int, Literal["mean", "min", "max"] | None, bool, int],
):
    # Helper function to create an array simulating an image stack
    def create_test_array(shape: tuple, frames: int, dtype: str) -> np.ndarray:
        for f in range(frames):
            # Increment array values by 1 per frame
            frame = np.ones(shape).astype(dtype) * f
            if f == 0:
                arr = np.array([frame])
            else:
                arr = np.append(arr, [frame], axis=0)
        return arr

    # Unpack test parameters
    img_type, frames, mode, is_float, result = test_params

    # Choose "int" and "float" dtypes
    dtype = "float64" if is_float is True else "int64"

    # Choose between grayscale or RGB image
    if img_type == "gray":
        shape: tuple[int, ...] = (64, 64)
    elif img_type == "rgb":
        shape = (64, 64, 3)
    else:
        raise ValueError("Unexpected value for image type")

    # Create image stack and flatten it
    arr = create_test_array(shape, frames, dtype)
    arr_new = flatten_image(arr, mode) if isinstance(mode, str) else flatten_image(arr)

    # Create new flattened array with the expected pixel value
    #   Because pixel values increase from 0 to (frame - 1) per frame, it's possible
    #   to predict what the array values will be after flattening using "mean", "max",
    #   or "min".
    arr_ref = np.ones(shape).astype(dtype) * result
    # DEBUG: Check that reference array has expected properties
    assert np.all(arr_ref == result) and arr_ref.shape == shape

    # Check that deviations are within a set threshold:
    # arr_1 - arr_0 = atol + rtol * abs(arr_0)
    np.testing.assert_allclose(
        arr_new,
        arr_ref,
        rtol=0,
        atol=1e-20,  # Really, there shouldn't be a difference
    )


image_flattening_fail_cases: tuple[tuple, ...] = (
    # Image type | Frames | Mode | Is float?
    # 0
    ("gray", 5, "uvuvwevwevwe", True),
    ("gray", 5, "onyetenyevwe", False),
    ("gray", 5, "ugwemubwem", True),
    ("rgb", 5, "osas", False),
    ("rgb", 5, 5, True),
    # 5
    ("rgb", 5, True, False),
    ("rgb", 5, [], True),
    ("rgb", 5, (), True),
    ("rgb", 5, {}, True),
    ("rgb", 5, set(), True),
)


@pytest.mark.parametrize("test_params", image_flattening_fail_cases)
def test_flatten_image_fails(test_params: tuple[str, int, Any, bool]):
    # Helper function to create an array simulating an image stack
    def create_test_array(shape: tuple, frames: int, dtype: str) -> np.ndarray:
        for f in range(frames):
            # Increment array values by 1 per frame
            frame = np.ones(shape).astype(dtype) * f
            if f == 0:
                arr = np.array([frame])
            else:
                arr = np.append(arr, [frame], axis=0)
        return arr

    with pytest.raises(ValueError):
        # Unpack parameters
        img_type, frames, mode, is_float = test_params
        shape = (64, 64) if img_type == "gray" else (64, 64, 3)
        dtype = "float64" if is_float is True else "int64"

        arr = create_test_array(shape, frames, dtype)
        flatten_image(arr, mode)


image_merging_test_matrix = (
    # Type | Num images | Frames | Is float? | Expected pixel value*
    #   * NOTE: np.round rounds to the nearest EVEN number for values
    #   EXACTLY halfway between rounded decimal values (e.g. 0.5
    #   rounds to 0, -1.5 rounds to -2, etc.).
    # Test grayscale stacks
    ("gray", 1, 5, False, 0),
    ("gray", 2, 5, True, 0.5),
    ("gray", 3, 5, True, 1),
    ("gray", 4, 5, False, 2),
    # Test RGB stacks
    ("rgb", 1, 5, True, 0),
    ("rgb", 2, 5, False, 0),
    ("rgb", 3, 5, False, 1),
    ("rgb", 4, 5, True, 1.5),
    # Test on images
    ("gray", 1, 1, False, 0),
    ("gray", 2, 1, True, 0.5),
    ("rgb", 3, 1, False, 1),
    ("rgb", 4, 1, True, 1.5),
)


@pytest.mark.parametrize("test_params", image_merging_test_matrix)
def test_merge_images(test_params: tuple[str, int, int, bool, int | float]):
    # Unpack test parameters
    img_type, num_imgs, frames, is_float, result = test_params

    # Select dtype
    dtype = "float64" if is_float is True else "int64"

    # Set frame shape based on grayscale or RGB image
    if img_type == "gray":
        shape: tuple[int, ...] = (64, 64)
    elif img_type == "rgb":
        shape = (64, 64, 3)
    else:
        raise ValueError("Unexpected value for image type")

    # Create list of images/stacks and merge them
    arr_list: np.ndarray | list[np.ndarray] = []
    for n in range(num_imgs):
        # Increment values by image/stack
        arr = np.array([np.ones(shape) for f in range(frames)]).astype(dtype) * n
        if num_imgs == 1:
            arr_list = arr
        else:
            arr_list.append(arr)
    # DEBUG: Check that arrays are generated correctly
    assert all(str(type(arr)) == str(np.ndarray) for arr in arr_list)
    composite = merge_images(arr_list)

    # Create a reference stack to compare against
    #   All values for a single image are the same and increment per image from
    #   0 - (num_img -1), so the expected average value of the final product can be
    #   quickly calculated
    arr_ref = np.array([np.ones(shape) for f in range(frames)]).astype(dtype) * result
    # DEBUG: Check that reference array has the expected shape
    assert arr_ref.shape == (frames, *shape)

    # Check that deviations are within a set threshold:
    # arr_1 - arr_0 = atol + rtol * abs(arr_0)
    np.testing.assert_allclose(
        composite,
        arr_ref,
        rtol=0,
        atol=1e-20,  # Really, there shouldn't be a difference
    )


image_merging_fail_cases = (
    # Image type | Num images | Same frames? | Same size? | Same dtype?
    ("gray", 2, False, True, False),
    ("gray", 3, True, True, False),
    ("gray", 4, False, True, True),
    ("rgb", 2, True, False, True),
    ("rgb", 3, False, False, True),
    ("rgb", 4, True, False, False),
)


@pytest.mark.parametrize("test_params", image_merging_fail_cases)
def test_merge_images_fails(test_params: tuple[str, int, bool, bool, bool]):
    def create_test_array(shape, frames, dtype):
        for f in range(frames):
            frame = np.ones(shape).astype(dtype)
            if f == 0:
                arr = np.array([frame])
            else:
                arr = np.append(arr, [frame], axis=0)
        return arr

    with pytest.raises(ValueError):
        # Unpack test_params
        img_type, num_images, frames, size, dtypes = test_params

        # Create list of test arrays
        arr_list: list[np.ndarray] = []
        for i in range(num_images):
            shape: tuple[int, ...] = (64, 64) if size is True else (64 + i, 64 + i)
            shape = (*shape, 3) if img_type == "rgb" else shape
            num_frames = 5 if frames is True else 1 + i
            dtype = f"uint{int(8 * (2**i))}" if dtypes is False else "uint8"
            arr = create_test_array(shape, num_frames, dtype)
            arr_list.append(arr)

        merge_images(arr_list)


def test_get_histogram():
    pass


def test_get_percentiles():
    pass


def test_load_and_convert_image():
    pass


def test_resize_tile():
    pass


def test_write_stack_to_tiff():
    pass


def test_write_stack_to_tiff_fails():
    pass
