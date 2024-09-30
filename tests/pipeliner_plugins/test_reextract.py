from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import starfile

from cryoemservices.pipeliner_plugins import reextract


@mock.patch("cryoemservices.pipeliner_plugins.reextract.mrcfile")
def test_extract_single_micrograph(mock_mrcfile, tmp_path):
    """Test the particle resizing"""
    mock_mrcfile.open().__enter__().data = np.random.random((40, 40))

    # Need inputs for the excluded region, grid, and positions without background
    bg_region = np.array(
        [[0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]],
        dtype=bool,
    )
    bg_size = len(np.where(bg_region)[0])
    grid_matrix = np.arange(16 * 3).reshape(16, 3)
    flat_positions_matrix = np.arange(3 * bg_size).reshape(3, bg_size)

    reextract.reextract_single_micrograph(
        motioncorr_name=tmp_path / "MotionCorr/job002/Movies/foil_hole.mrc",
        reextract_name=tmp_path / "Extract/job025/Movies/foil_hole.mrcs",
        particles_x=[41.21, 9.82],
        particles_y=[40.32, 39.36],
        full_extract_width=4,
        scaled_extract_width=2,
        scaled_boxsize=4,
        scaled_pixel_size=1.026,
        grid_matrix=grid_matrix,
        flat_positions_matrix=flat_positions_matrix,
        bg_region=bg_region,
        invert_contrast=True,
        normalise=True,
        downscale=True,
    )

    mock_mrcfile.new().__enter__().set_data.assert_called_once()

    # Read in the output array and check it is correct within rounding errors
    new_mrc_array = mock_mrcfile.new().__enter__().set_data.call_args_list[0][0][0]
    assert new_mrc_array.shape == (2, 4, 4)
    assert (
        new_mrc_array[0][0].astype(np.float64).round(2) == [-1.76, -1.54, -1.32, -1.10]
    ).all()
    assert (
        new_mrc_array[0][1].astype(np.float64).round(2) == [-0.88, -0.66, -0.44, -0.22]
    ).all()
    assert (
        new_mrc_array[0][2].astype(np.float64).round(2) == [-0.00, 0.22, 0.44, 0.66]
    ).all()
    assert (
        new_mrc_array[0][3].astype(np.float64).round(2) == [0.88, 1.10, 1.32, 1.54]
    ).all()
    assert (
        new_mrc_array[1][0].astype(np.float64).round(2) == [-1.76, -1.54, -1.32, -1.10]
    ).all()
    assert (
        new_mrc_array[1][1].astype(np.float64).round(2) == [-0.88, -0.66, -0.44, -0.22]
    ).all()
    assert (
        new_mrc_array[1][2].astype(np.float64).round(2) == [-0.00, 0.22, 0.44, 0.66]
    ).all()
    assert (
        new_mrc_array[1][3].astype(np.float64).round(2) == [0.88, 1.10, 1.32, 1.54]
    ).all()

    # Check the mrc header
    assert mock_mrcfile.new().__enter__().header.mx == 4
    assert mock_mrcfile.new().__enter__().header.my == 4
    assert mock_mrcfile.new().__enter__().header.mz == 1
    assert mock_mrcfile.new().__enter__().header.cella.x == 4 * 1.026
    assert mock_mrcfile.new().__enter__().header.cella.y == 4 * 1.026
    assert mock_mrcfile.new().__enter__().header.cella.z == 1


@mock.patch("cryoemservices.pipeliner_plugins.reextract.ProcessPoolExecutor")
def test_reextract_run(mock_pool, tmp_path):
    """Test the command line particle reextraction function"""

    # Make the expected particles input file
    (tmp_path / "Select/job009").mkdir(parents=True)
    with open(f"{tmp_path}/Select/job009/particles.star", "w") as particles_file:
        particles_file.write(
            "data_optics\nloop_\n"
            "_dummy1\n_dummy2\n_dummy3\n_dummy4\n_dummy5\n_dummy6\n"
            "_ImagePixelSize7\n_ImageSize8\nopticsGroup1 x x x x x 2.565 48\n\n"
            "data_particles\nloop_\n_CoordinateX1\n_CoordinateY2\n"
            "_ImageName3\n_MicrographName4\n"
            "_dummy5\n_dummy6\n_dummy7\n_dummy8\n_dummy9\n_dummy10\n_dummy11\n"
            "_dummy12\n_dummy13\n_dummy14\n_dummy15\n_dummy16\n_dummy17\n"
            "_OriginXAngst18\n_OriginYAngst19\n"
            "4133.389320 4005.140464 2@Extract/job008/Movies/FoilHole.mrcs "
            "MotionCorr/job002/Movies/FoilHole.mrc x x x x x x x x x x x x x "
            "6.451806 -14.06819\n"
            "994.023314 3914.861541 3@Extract/job008/Movies/FoilHole.mrcs "
            "MotionCorr/job002/Movies/FoilHole.mrc x x x x x x x x x x x x x "
            "6.451807 -11.50319\n"
        )

    # Set the command line arguments
    sys.argv = [
        "reextract",
        "--extract_job_dir",
        f"{tmp_path}/Extract/job025",
        "--select_job_dir",
        f"{tmp_path}/Select/job009",
        "--original_dir",
        str(tmp_path),
        "--full_boxsize",
        "248",
        "--scaled_boxsize",
        "124",
        "--full_pixel_size",
        "0.513",
        "--scaled_pixel_size",
        "1.026",
        "--bg_radius",
        "-1",
        "--invert_contrast",
        "--normalise",
        "--downscale",
    ]
    reextract.run()

    # Check the output star file fields
    assert (tmp_path / "Extract/job025/particles.star").is_file()
    extracted_particles = starfile.read(tmp_path / "Extract/job025/particles.star")
    assert extracted_particles["optics"]["ImagePixelSize7"].values == [1.026]
    assert extracted_particles["optics"]["ImageSize8"].values == [124]
    assert (
        extracted_particles["particles"]["ImageName3"].values[0]
        == "000001@Extract/job025/Movies/FoilHole.mrcs"
    )
    assert (
        extracted_particles["particles"]["ImageName3"].values[1]
        == "000002@Extract/job025/Movies/FoilHole.mrcs"
    )
    assert extracted_particles["particles"]["CoordinateX1"].values[0] == 4121.38932
    assert extracted_particles["particles"]["CoordinateX1"].values[1] == 982.023314
    assert extracted_particles["particles"]["CoordinateY2"].values[0] == 4032.140464
    assert extracted_particles["particles"]["CoordinateY2"].values[1] == 3936.861541
    assert extracted_particles["particles"]["OriginXAngst18"].values[0] == -5.548194
    assert extracted_particles["particles"]["OriginXAngst18"].values[1] == -5.548193
    assert extracted_particles["particles"]["OriginYAngst19"].values[0] == 12.93181
    assert extracted_particles["particles"]["OriginYAngst19"].values[1] == 10.49681

    assert (tmp_path / "Extract/job025/extractpick.star").is_file()
    extract_picks = starfile.read(tmp_path / "Extract/job025/extractpick.star")
    assert len(extract_picks["rlnMicrographName"].values) == 1
    assert (
        extract_picks["rlnMicrographName"].values[0]
        == "MotionCorr/job002/Movies/FoilHole.mrc"
    )
    assert (
        extract_picks["rlnMicrographCoordinates"].values[0]
        == "Extract/job025/Movies/FoilHole.mrcs"
    )

    # Check the particle extraction function is called
    assert mock_pool().__enter__().submit.call_count == 1
    mock_pool().__enter__().submit.assert_called_with(
        mock.ANY,
        motioncorr_name=tmp_path / "MotionCorr/job002/Movies/FoilHole.mrc",
        reextract_name=tmp_path / "Extract/job025/Movies/FoilHole.mrcs",
        particles_x=[4121.38932, 982.023314],
        particles_y=[4032.140464, 3936.861541],
        full_extract_width=124,
        scaled_extract_width=62,
        scaled_boxsize=124,
        scaled_pixel_size=1.026,
        grid_matrix=mock.ANY,
        flat_positions_matrix=mock.ANY,
        bg_region=mock.ANY,
        invert_contrast=True,
        normalise=True,
        downscale=True,
    )
