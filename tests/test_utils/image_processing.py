from typing import Any

import cv2
import numpy as np


def gaussian_2d(
    shape: tuple[int, int],
    amplitude: int | float,
    centre: tuple[int, int],
    sigma: tuple[float, float],
    theta: float,
    offset: int | float,
):
    """
    Helper function to create Gaussian peaks.

    Parameters
    ----------
    shape: tuple[int, int]
        The shape of the image frame to place the 2D Gaussian in

    amplitude: int | float
        Amplitude of the Gaussian

    centre: tuple[int, int]
        The x- and y-coordinates around which the Gaussian is based

    sigma: tuple[float, float]
        The length of the major and minor axes

    theta: float
        The angle of the Gaussian, in degrees

    offset: int | float
        The intensity offset of the Gaussian
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


def create_grayscale_image_with_gaussian(
    shape: tuple[int, int],
    num_frames: int,
    dtype: str,
    peaks: list[dict[str, Any]],
    peak_shift_per_frame: tuple[int, int],
    intensity_offset_per_frame: int,
):
    """
    Creates a grayscale image with Gaussian peak features that are offset from
    frame-to-frame.

    Parameters
    ----------
    shape: tuple[int, int]
        The shape of the image frame

    num_frames: int
        The number of frames in the image stack

    dtype: str
        The dtype of the image

    peaks: list[dict]
        A list of kwargs that will be passed to the 2D Gaussian creation function
        to populate the image. The dictionaries should contain the following keys:
            - shape: tuple[int, int]
            - amplitude: int
            - centre: tuple[int, int]
            - sigma: tuple[float, float]
            - theta: float
            - offset: int | float

    """

    x_shift, y_shift = peak_shift_per_frame
    c_off = intensity_offset_per_frame

    arr = np.zeros((num_frames, *shape), dtype=dtype)
    for f in range(num_frames):
        for peak in peaks:
            arr[f] += gaussian_2d(**peak).astype(dtype)
            # Adjust the peak offset per frame after updating the current one
            centre: tuple[int, int] = peak["centre"]
            x, y = centre
            peak["centre"] = (x + x_shift, y + y_shift)
            peak["offset"] += c_off

    arr = arr.astype(dtype)
    if num_frames == 1:
        arr = arr[0]

    return arr


def create_grayscale_image_with_holes(
    shape: tuple[int, int],
    num_frames: int,
    layer_intensity: int,
    noise_sigma: int,
    hole_intensity: int,
    hole_list: list[dict[str, Any]],
    offset_per_frame: tuple[int, int],
):
    """
    Creates a textured test image with hole-like features in it to test the
    ORB-based image registration function with.

    Parameters
    ----------
    shape: tuple[int, int]
        The shape of the image frame

    num_frames: int
        The number of frames in the image stack

    layer_intensity: int
        The pixel value to use for the simulated layer in the image

    noise_sigma: int
        The amount of Gaussian noise to introduce into the image

    hole_intensity: int
        The pixel value to use to represent the hole

    hole_list: list[dict[str, Any]]
        A list of dictionaries used to generate the holes needed for this
        test image. The needed parameters are:
        - center: tuple[int, int] |
            The x and y coordinates of the hole
        - axes: tuple[int, int] |
            The lengths of the major and minor axes
        - angle: float |
            The angle of the hole, in degrees

    offset_per_frame: tuple[int, int]
        The x and y offset to shift the image by each frame
    """

    # Create a noisy array
    arr = np.ones((num_frames, *shape), dtype=np.float32) * layer_intensity
    arr += np.random.normal(0, noise_sigma, (num_frames, *shape))

    # Iterate and create holes
    x_offset, y_offset = offset_per_frame
    for f in range(num_frames):
        for hole_info in hole_list:
            center: tuple[int, int] = hole_info["center"]
            cx, cy = center
            axes: tuple[int, int] = hole_info["axes"]
            ax, ay = axes

            arr[f] = cv2.ellipse(
                arr[f],
                center=(cx + (x_offset * f), cy + (y_offset * f)),
                axes=(ax, ay),
                angle=hole_info["angle"],
                startAngle=0,
                endAngle=360,
                color=hole_intensity,
                thickness=-1,
            )
        # Apply slight Gaussian blurring
        arr[f] = cv2.GaussianBlur(arr[f], (0, 0), 1.0)

    np.clip(arr, 0, 255, out=arr)
    arr = arr.astype(np.uint8)
    if num_frames == 1:
        arr = arr[0]
    return arr
