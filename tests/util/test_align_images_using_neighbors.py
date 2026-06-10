from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from cryoemservices.util.align_images_using_neighbors import (
    AnnotationParameters,
    _determine_annotation_parameters,
    align_images_using_neighbors,
)
from tests.test_utils.image_processing import create_grayscale_image_with_holes


@pytest.mark.parametrize(
    "test_params",
    (  # Input shape | Expected annotation params
        ((5000, 5000), (4, 5, 2.0, 32)),
        ((4000, 4000), (3, 4, 1.5, 24)),
        ((2000, 2000), (2, 3, 1.0, 16)),
        ((1000, 1000), (1, 3, 0.8, 12)),
        ((500, 500), (1, 3, 0.6, 10)),
        ((200, 200), (1, 2, 0.4, 8)),
    ),
)
def test_determine_annotation_parameters(
    test_params: tuple[tuple[int, int], tuple[int, int, float, int]],
):
    # Unpack test params
    shape, (line_thickness, marker_size, font_scale, text_offset) = test_params

    # Create a mock array with a 'shape' attribute
    mock_img = MagicMock()
    mock_img.shape = shape

    # Verify that the expected parameters were selected
    assert _determine_annotation_parameters(mock_img) == AnnotationParameters(
        line_thickness=line_thickness,
        marker_size=marker_size,
        font_scale=font_scale,
        text_offset=text_offset,
    )


@pytest.mark.parametrize(
    "test_params",
    (  # x-offset | y-offset
        (10, 10),
        (10, -10),
        (-10, 10),
        (-10, -10),
    ),
)
def test_align_images_using_neighbors(
    test_params,
    tmp_path: Path,
):
    # Set parameters
    w, h = 512, 512
    x_offset, y_offset = test_params

    # Create grayscale image with hole-like features
    ref = create_grayscale_image_with_holes(
        shape=(h, w),
        num_frames=1,
        layer_intensity=128,
        noise_sigma=2,
        hole_intensity=16,
        hole_list=[
            {
                "center": (140, 320),
                "axes": (48, 16),
                "angle": 90,
            },
            {
                "center": (160, 190),
                "axes": (40, 16),
                "angle": 90,
            },
            {
                "center": (250, 340),
                "axes": (48, 20),
                "angle": 90,
            },
            {
                "center": (280, 150),
                "axes": (32, 12),
                "angle": 90,
            },
            {
                "center": (290, 240),
                "axes": (24, 12),
                "angle": 90,
            },
            {
                "center": (360, 170),
                "axes": (32, 12),
                "angle": 90,
            },
            {
                "center": (360, 330),
                "axes": (40, 12),
                "angle": 90,
            },
        ],
        offset_per_frame=(0, 0),
    )
    mov = create_grayscale_image_with_holes(
        shape=(h, w),
        num_frames=1,
        layer_intensity=128,
        noise_sigma=2,
        hole_intensity=16,
        hole_list=[
            {
                "center": (140 + x_offset, 320 + y_offset),
                "axes": (48, 16),
                "angle": 90,
            },
            {
                "center": (160 + x_offset, 190 + y_offset),
                "axes": (40, 16),
                "angle": 90,
            },
            {
                "center": (250 + x_offset, 340 + y_offset),
                "axes": (48, 20),
                "angle": 90,
            },
            {
                "center": (280 + x_offset, 150 + y_offset),
                "axes": (32, 12),
                "angle": 90,
            },
            {
                "center": (290 + x_offset, 240 + y_offset),
                "axes": (24, 12),
                "angle": 90,
            },
            {
                "center": (360 + x_offset, 170 + y_offset),
                "axes": (32, 12),
                "angle": 90,
            },
            {
                "center": (360 + x_offset, 330 + y_offset),
                "axes": (40, 12),
                "angle": 90,
            },
        ],
        offset_per_frame=(0, 0),
    )
    result = align_images_using_neighbors(
        ref,
        mov,
        median_blur=3,
        gaussian_blur=0.5,
        sobel_kernel=3,
        use_hanning=True,
        min_component_area=20,
        threshold_percentile=98,
        morph_close_kernel=16,
        morph_open_kernel=2,
        min_feature_area=20,
        max_feature_area=2500,
        min_solidity=0.6,
        min_ellipse_fit=0.4,
        max_aspect_ratio=0.9,
        max_neighbor_distance=200,
        min_score=0.4,
        ransac_threshold=5,
        save_tables=True,
        save_images=True,
        save_dir=tmp_path,
    )
    aln = result.get("aligned", None)
    assert isinstance(aln, np.ndarray)

    # Compare only the common region
    w0, h0 = abs(x_offset), abs(y_offset)
    w1, h1 = w - w0, h - h0
    expected = ref[h0:h1, w0:w1]
    returned = aln[h0:h1, w0:w1]

    # Check that the number of pixels that image alignment mismatch is within bounds
    mismatch = abs(expected.astype(np.float32) - returned.astype(np.float32)) > 5
    assert np.sum(mismatch) / np.prod(mismatch.shape) < 0.005  # 0.5% pixel deviation

    # Assert that the expected intermediate files were produced
    for name in ("ref", "mov"):
        for stub in (
            "desc.tsv",
            "features.png",
            "features.tsv",
            "filled.png",
            "filtered.png",
            "gaussian.png",
            "hanning.png",
            "median.png",
            "sobel.png",
            "threshold.png",
        ):
            assert (tmp_path / f"{name}_{stub}").exists()
    assert (tmp_path / "matched_holes.png").exists()
    assert (tmp_path / "overlay.png").exists()
    assert (tmp_path / "scores.tsv").exists()
