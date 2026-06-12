from typing import Literal

import numpy as np
import pytest

from cryoemservices.util.drift_correct_image import drift_correct_image
from tests.test_utils.image_processing import create_grayscale_image_with_gaussian


@pytest.mark.parametrize(
    "test_params",
    (
        # x-offset | y-offset | Alignment starting point
        # Test uint8 arrays
        (1, 1, "beginning"),
        (1, 1, "middle"),
        (1, 1, "end"),
        (2, 2, "beginning"),
        (2, 2, "middle"),
        (2, 2, "end"),
        (-1, -1, "beginning"),
        (-1, -1, "middle"),
        (-1, -1, "end"),
        (-2, -2, "beginning"),
        (-2, -2, "middle"),
        (-2, -2, "end"),
    ),
)
def test_drift_correct_image(
    test_params: tuple[int, int, Literal["beginning", "middle", "end"]],
):
    # Unpack test params
    x_offset, y_offset, start_point = test_params
    num_frames = 10
    shape = (128, 128)
    dtype = "uint8"

    arr = create_grayscale_image_with_gaussian(
        shape=shape,
        num_frames=num_frames,
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
        peak_shift_per_frame=(x_offset, y_offset),
        intensity_offset_per_frame=0,
    )

    # Align the frames in the stack
    aligned = drift_correct_image(array=arr, start_from=start_point)

    # Assert that bright spots are aligned throughout the stack
    for f in range(num_frames):
        if f == 0:
            continue
        np.testing.assert_allclose(
            aligned[f - 1],
            aligned[f],
            rtol=1,
            atol=5,
            # Absolute intensities can change by quite a bit during alignment
        )
