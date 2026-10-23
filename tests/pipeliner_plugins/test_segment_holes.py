import subprocess
import sys
from unittest import mock

import mrcfile
import numpy as np

from cryoemservices.pipeliner_plugins import segment_holes


@mock.patch("cryoemservices.pipeliner_plugins.segment_holes.set_up_sam")
def test_segment_single_image_all_hole(mock_sam, tmp_path):
    """Test simplest case of segmentation, all hole"""
    mock_generator = mock.Mock()
    mock_generator.generate.return_value = [
        {"area": 100 * 100, "segmentation": np.ones((100, 100), dtype=bool)}
    ]
    mock_sam.return_value = mock_generator

    with mrcfile.new(tmp_path / "test.mrc", overwrite=True) as mrc:
        mrc.set_data((np.random.rand(100, 100) * 255).astype(dtype=np.uint8))

    im_results, mask_results = segment_holes.segment_single_image(
        impath=tmp_path / "test.mrc",
        checkpoint=tmp_path / "checkpoint.ckpt",
        output_dir=tmp_path,
        binning=1,
        min_area_fraction=0.01,
        beam_threshold=10,
        hole_threshold=100,
        save_summary=True,
    )

    # Check SAM calls
    mock_sam.assert_called_once_with(tmp_path / "checkpoint.ckpt")
    mock_generator.generate.assert_called_once()

    # Check return structure
    assert im_results.get("hole_percent") == 100.0
    assert im_results.get("hole_detected", False)
    assert not im_results.get("beam_edge_detected", True)
    assert im_results.get("n_masks") == 1

    assert len(mask_results) == 1
    assert mask_results[0].get("area_fraction") == 100.0
    assert mask_results[0].get("mask_area") == 100 * 100

    # Check output images were made
    assert (tmp_path / "test_mask.png").is_file()
    assert (tmp_path / "test_overlay.png").is_file()


@mock.patch("cryoemservices.pipeliner_plugins.segment_holes.set_up_sam")
def test_segment_single_image_hole_and_beam(mock_sam, tmp_path):
    """Test case of segmentation with a beam (values < 10) present"""
    mock_generator = mock.Mock()
    mask = np.ones((100, 100), dtype=bool)
    for i in range(10):
        mask[i] = False
    mock_generator.generate.return_value = [
        {"area": 100 * 90, "segmentation": mask},
        {"area": 100 * 10, "segmentation": ~mask},
    ]
    mock_sam.return_value = mock_generator

    image = (np.random.rand(100, 100) * 255).astype(dtype=np.uint8)
    for i in range(10):
        image[i] = 1

    with mrcfile.new(tmp_path / "test.mrc", overwrite=True) as mrc:
        mrc.set_data(image)

    im_results, mask_results = segment_holes.segment_single_image(
        impath=tmp_path / "test.mrc",
        checkpoint=tmp_path / "checkpoint.ckpt",
        output_dir=tmp_path,
        binning=1,
        min_area_fraction=0.01,
        beam_threshold=10,
        hole_threshold=100,
        save_summary=True,
    )

    # Check SAM calls
    mock_sam.assert_called_once_with(tmp_path / "checkpoint.ckpt")
    mock_generator.generate.assert_called_once()

    # Check return structure
    assert im_results.get("hole_percent") == 90.0
    assert im_results.get("hole_detected", False)
    assert im_results.get("beam_edge_detected", False)
    assert im_results.get("n_masks") == 2

    assert len(mask_results) == 2
    assert mask_results[0].get("area_fraction") == 90.0
    assert mask_results[0].get("mask_area") == 100 * 90
    assert mask_results[1].get("area_fraction") == 10.0
    assert mask_results[1].get("mask_area") == 100 * 10

    # Check output images were made
    assert (tmp_path / "test_mask.png").is_file()
    assert (tmp_path / "test_overlay.png").is_file()


@mock.patch("cryoemservices.pipeliner_plugins.segment_holes.segment_single_image")
def test_run_segment_holes_plot(mock_segment, tmp_path):
    """Test that the cli can call the segment function"""
    mock_segment.return_value = (
        {
            "image": "image_name",
            "hole_percent": 100,
            "hole_detected": True,
            "beam_edge_detected": False,
            "hole_median_intensity": 100,
            "n_masks": 1,
        },
        [
            {
                "image": "image_name",
                "mask_id": 0,
                "mask_area": 1,
                "median_intensity": 100,
                "area_fraction": 100.0,
            }
        ],
    )

    (tmp_path / "image.mrc").touch()

    sys.argv = [
        "cryoemservices.segment_holes",
        "--micrograph_dir",
        str(tmp_path),
        "--input_glob",
        "*.mrc",
        "--output_dir",
        str(tmp_path),
        "--checkpoint",
        f"{tmp_path}/checkpoint.ckpt",
    ]
    segment_holes.run()

    mock_segment.assert_called_once_with(
        impath=tmp_path / "image.mrc",
        checkpoint=tmp_path / "checkpoint.ckpt",
        output_dir=tmp_path,
        binning=4,
        min_area_fraction=0.001,
        beam_threshold=10,
        hole_threshold=100,
        save_summary=False,
    )

    assert (tmp_path / "hole_percentages.csv").is_file()
    assert (tmp_path / "mask_statistics.csv").is_file()


def test_segment_cli_exists():
    """Test the hole segmentation CLI is made"""
    result = subprocess.run(
        [
            "cryoemservices.segment_holes",
            "--help",
        ],
        capture_output=True,
    )
    assert not result.returncode

    # Find the first line of the help and strip out all the spaces and newlines
    stdout_as_string = result.stdout.decode("utf8", "replace")
    cleaned_help_line = (
        stdout_as_string.split("\n\n")[0].replace("\n", "").replace(" ", "")
    )
    assert cleaned_help_line == (
        "usage:cryoemservices.segment_holes[-h]"
        "--micrograph_dirMICROGRAPH_DIR--input_globINPUT_GLOB--output_dirOUTPUT_DIR"
        "--checkpointCHECKPOINT[--binningBINNING][--min_area_fractionMIN_AREA_FRACTION]"
        "[--beam_thresholdBEAM_THRESHOLD][--hole_thresholdHOLE_THRESHOLD][--save_summary]"
    )
