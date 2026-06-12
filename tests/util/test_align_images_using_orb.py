import numpy as np
import pytest

from cryoemservices.util.align_images_using_orb import align_images_using_orb
from tests.test_utils.image_processing import create_grayscale_image_with_holes


@pytest.mark.parametrize(
    "test_params",
    (  # Num frames | x-offset | y-offset
        (1, 20, 20),
        (5, 20, 20),
        (1, -10, -10),
        (5, -10, -10),
        (1, 20, -20),
        (5, -20, 20),
        (1, -10, 10),
        (5, 10, -10),
    ),
)
def test_align_images_using_orb(test_params: tuple[int, int, int]):
    # Unpack test params
    num_frames, x_offset, y_offset = test_params

    # Construct the grayscale images to align together
    w, h = 512, 512
    ref = create_grayscale_image_with_holes(
        shape=(h, w),
        num_frames=num_frames,
        layer_intensity=96,
        noise_sigma=4,
        hole_intensity=32,
        hole_list=[
            {
                "center": (140, 320),
                "axes": (24, 4),
                "angle": 90,
            },
            {
                "center": (160, 190),
                "axes": (20, 4),
                "angle": 90,
            },
            {
                "center": (250, 340),
                "axes": (24, 4),
                "angle": 90,
            },
            {
                "center": (280, 150),
                "axes": (16, 4),
                "angle": 90,
            },
            {
                "center": (290, 240),
                "axes": (12, 4),
                "angle": 90,
            },
            {
                "center": (360, 170),
                "axes": (16, 4),
                "angle": 90,
            },
            {
                "center": (360, 330),
                "axes": (20, 4),
                "angle": 90,
            },
        ],
        offset_per_frame=(0, 0),
    )
    mov = create_grayscale_image_with_holes(
        shape=(512, 512),
        num_frames=num_frames,
        layer_intensity=96,
        noise_sigma=4,
        hole_intensity=32,
        hole_list=[
            {
                "center": (140 + x_offset, 320 + y_offset),
                "axes": (24, 4),
                "angle": 90,
            },
            {
                "center": (160 + x_offset, 190 + y_offset),
                "axes": (20, 4),
                "angle": 90,
            },
            {
                "center": (250 + x_offset, 340 + y_offset),
                "axes": (24, 4),
                "angle": 90,
            },
            {
                "center": (280 + x_offset, 150 + y_offset),
                "axes": (16, 4),
                "angle": 90,
            },
            {
                "center": (290 + x_offset, 240 + y_offset),
                "axes": (12, 4),
                "angle": 90,
            },
            {
                "center": (360 + x_offset, 170 + y_offset),
                "axes": (16, 4),
                "angle": 90,
            },
            {
                "center": (360 + x_offset, 330 + y_offset),
                "axes": (20, 4),
                "angle": 90,
            },
        ],
        offset_per_frame=(0, 0),
    )
    aln = align_images_using_orb(
        ref,
        mov,
        sigma=3,
        hanning_window=True,
        kernel_size=3,
        min_area=50,
        max_area=5000,
        patch_size=31,
    )
    # Compare only the common region
    w0, h0 = abs(x_offset), abs(y_offset)
    w1, h1 = w - w0, h - h0
    expected = ref[:, h0:h1, w0:w1] if num_frames != 1 else ref[h0:h1, w0:w1]
    returned = aln[:, h0:h1, w0:w1] if num_frames != 1 else aln[h0:h1, w0:w1]

    # # Check that the number of pixels that image alignment mismatch is within bounds
    mismatch = abs(expected.astype(np.float32) - returned.astype(np.float32)) > 5
    assert np.sum(mismatch) / np.prod(mismatch.shape) < 0.005  # 0.5% pixel deviation
