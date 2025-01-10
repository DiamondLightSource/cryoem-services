"""
Recurring helper functions used when generating the fixtures needed to test the
CLEM workflow.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def gaussian_2d(
    shape: tuple[int, int],
    amplitude: float,
    centre: tuple[int, int],
    sigma: tuple[float, float],
    theta: float,
    offset: float,
):
    """
    Helper function to create Gaussian peaks
    """
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
        amplitude * np.exp(-(x_rot**2 / (2 * sig_x**2)) - (y_rot**2 / (2 * sig_y**2)))
        + offset
    )

    return gaussian


def create_grayscale_image(
    shape: tuple[int, int],
    num_frames: int,
    dtype: str,
    peaks: list[dict[str, Any]],
    peak_offset_per_frame: tuple[int, int],
    intensity_offset_per_frame: int,
):
    """
    Creates a grayscale image with peaks that are offset from frame-to-frame
    """

    x_off, y_off = peak_offset_per_frame
    c_off = intensity_offset_per_frame
    if num_frames == 1:
        arr = np.zeros(shape, dtype=dtype)
        for peak in peaks:
            arr += gaussian_2d(**peak).astype(dtype)
    else:
        arr = np.zeros((num_frames, *shape), dtype=dtype)
        for f in range(num_frames):
            for peak in peaks:
                # Adjust the peak offset per frame
                centre: tuple[int, int] = peak["centre"]
                x, y = centre
                peak["centre"] = (x + (f * x_off), y + (f * y_off))
                peak["offset"] += f * c_off
                arr[f] += gaussian_2d(**peak).astype(dtype)

    arr = arr.astype(dtype)

    return arr
