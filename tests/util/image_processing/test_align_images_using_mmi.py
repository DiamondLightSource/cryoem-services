import numpy as np
import pytest

from cryoemservices.util.image_processing.align_images_using_mmi import (
    align_images_using_mmi,
)
from tests.test_utils.image_processing import create_grayscale_image_with_gaussian


@pytest.mark.parametrize(
    "test_params",
    (
        # x-offset | y-offset | dtype
        (1, 1, "uint8"),
        (2, 2, "uint8"),
        (-1, 1, "uint8"),
        (-2, 2, "uint8"),
    ),
)
def test_align_images_using_mmi(test_params: tuple[int, int, str]):
    # Unpack test params
    x_offset, y_offset, dtype = test_params

    n_frames = 5
    shape = (128, 128)

    ref = create_grayscale_image_with_gaussian(
        shape=shape,
        num_frames=n_frames,
        dtype=dtype,
        peaks=[
            {
                "shape": shape,
                "amplitude": 192,
                "centre": (48, 48),
                "sigma": (8, 12),
                "theta": 30,
                "offset": 0,
            },
            {
                "shape": shape,
                "amplitude": 128,
                "centre": (80, 80),
                "sigma": (12, 8),
                "theta": 45,
                "offset": 0,
            },
        ],
        peak_shift_per_frame=(0, 0),
        intensity_offset_per_frame=0,
    )
    mov = create_grayscale_image_with_gaussian(
        shape=shape,
        num_frames=n_frames,
        dtype=dtype,
        peaks=[
            {
                "shape": shape,
                "amplitude": 192,
                "centre": (48 + x_offset, 48 + y_offset),
                "sigma": (8, 12),
                "theta": 30,
                "offset": 0,
            },
            {
                "shape": shape,
                "amplitude": 128,
                "centre": (80 + x_offset, 80 + y_offset),
                "sigma": (12, 8),
                "theta": 45,
                "offset": 0,
            },
        ],
        peak_shift_per_frame=(0, 0),
        intensity_offset_per_frame=0,
    )

    # Align moving image to reference
    reg = align_images_using_mmi(ref, mov)

    # Assert that bright spots are aligned
    np.testing.assert_allclose(
        ref,
        reg,
        rtol=1,
        atol=5,
        # Absolute intensities can change by quite a bit during alignment
    )
