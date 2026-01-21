from __future__ import annotations

from pathlib import Path
from typing import Optional
from unittest import mock

import mrcfile
import numpy as np
import PIL.Image
import pytest
import tifffile as tf

from cryoemservices.services.images_plugins import (
    mrc_central_slice,
    mrc_projection,
    mrc_to_apng,
    mrc_to_apng_colour,
    mrc_to_jpeg,
    picked_particles,
    picked_particles_3d_apng,
    picked_particles_3d_central_slice,
    tiff_to_apng,
    tilt_series_alignment,
)
from cryoemservices.util.clem_array_functions import convert_to_rgb


def plugin_params(
    jpeg_path: Path,
    all_frames: bool,
    pixel_size: float = 0,
    projection: str = "XY",
    thickness: float = 0,
):
    def params(key):
        p = {
            "parameters": {"images_command": "mrc_to_jpeg"},
            "file": jpeg_path.with_suffix(".mrc"),
            "aln_file": jpeg_path.with_suffix(".aln"),
            "all_frames": all_frames,
            "pixel_spacing": pixel_size,
            "pixel_size": pixel_size,
            "projection": projection,
            "thickness_ang": thickness,
        }
        return p.get(key)

    return params


def plugin_params_central(jpeg_path, skip_rescaling=False, jitter_edge=False):
    def params(key):
        p = {
            "parameters": {"images_command": "mrc_central_slice"},
            "file": jpeg_path.with_suffix(".mrc"),
            "skip_rescaling": skip_rescaling,
            "jitter_edge": jitter_edge,
        }
        return p.get(key)

    return params


def plugin_params_apng_colour(file_list: list[str], mask: Optional[str] = None):
    def params(key):
        p = {
            "parameters": {"images_command": "mrc_to_apng_colour"},
            "file_list": file_list,
            "outfile": str(Path(file_list[0]).with_suffix("")) + "_movie.png",
            "mask": mask,
        }
        return p.get(key)

    return params


def plugin_params_parpick(jpeg_path, outfile):
    def params(key, default=None):
        p = {
            "parameters": {"images_command": "picked_particles"},
            "file": jpeg_path,
            "coordinates": [("0", "1"), ("2", "2")],
            "angpix": 0.5,
            "diameter": 190,
            "outfile": outfile,
            "flatten_image": True,
        }
        return p.get(key) or default

    return params


def plugin_params_tomo_pick(input_file, coords_file, command):
    def params(key):
        p = {
            "parameters": {"images_command": command},
            "file": input_file,
            "coordinates_file": coords_file,
            "box_size": 40,
        }
        return p.get(key)

    return params


def test_mrc_to_jpeg_nack_when_file_not_found(tmp_path):
    jpeg_path = tmp_path / "new_folder/new_job/new_file.jpeg"
    assert not mrc_to_jpeg(plugin_params(jpeg_path, False))


def test_mrc_to_jpeg_2d_ack_when_file_exists(tmp_path):
    jpeg_path = tmp_path / "convert_test.jpeg"
    test_data = np.arange(9, dtype=np.int8).reshape(3, 3)
    with mrcfile.new(jpeg_path.with_suffix(".mrc")) as mrc:
        mrc.set_data(test_data)
    assert mrc_to_jpeg(plugin_params(jpeg_path, False)) == jpeg_path
    assert jpeg_path.is_file()


def test_mrc_to_jpeg_2d_ack_with_scalebar(tmp_path):
    jpeg_path = tmp_path / "convert_test.jpeg"
    test_data = np.arange(9, dtype=np.int8).reshape(3, 3)
    with mrcfile.new(jpeg_path.with_suffix(".mrc")) as mrc:
        mrc.set_data(test_data)
    assert mrc_to_jpeg(plugin_params(jpeg_path, False, 1.2)) == jpeg_path
    assert jpeg_path.is_file()


def test_mrc_to_jpeg_3d_ack_all_frames(tmp_path):
    """All frames version makes a file for every frame of the movie"""
    jpeg_path = tmp_path / "convert_test.jpeg"
    test_data = np.arange(27, dtype=np.int8).reshape((3, 3, 3))
    with mrcfile.new(jpeg_path.with_suffix(".mrc")) as mrc:
        mrc.set_data(test_data)
    assert mrc_to_jpeg(plugin_params(jpeg_path, True)) == [
        tmp_path / f"{jpeg_path.stem}_{i}.jpeg" for i in range(1, 4)
    ]
    for i in range(1, 4):
        assert (tmp_path / f"{jpeg_path.stem}_{i}.jpeg").is_file()


def test_mrc_to_jpeg_3d_ack_not_all_frames(tmp_path):
    """Only make an image from the first frame"""
    jpeg_path = tmp_path / "convert_test.jpeg"
    test_data = np.arange(27, dtype=np.int8).reshape((3, 3, 3))
    with mrcfile.new(jpeg_path.with_suffix(".mrc")) as mrc:
        mrc.set_data(test_data)
    assert mrc_to_jpeg(plugin_params(jpeg_path, False)) == jpeg_path
    assert (tmp_path / "convert_test.jpeg").is_file()


def test_picked_particles_processes_when_basefile_exists(tmp_path):
    base_mrc_path = tmp_path / "base.mrc"
    out_jpeg_path = tmp_path / "processed.jpeg"
    test_data = np.arange(16, dtype=np.int8).reshape(4, 4)
    with mrcfile.new(base_mrc_path) as mrc:
        mrc.set_data(test_data)
    assert (
        picked_particles(plugin_params_parpick(base_mrc_path, out_jpeg_path))
        == out_jpeg_path
    )


def test_picked_particles_fails_when_basefile_does_not_exist(tmp_path):
    base_mrc_path = tmp_path / "base.mrc"
    out_jpeg_path = tmp_path / "processed.jpeg"
    assert not picked_particles(plugin_params_parpick(base_mrc_path, out_jpeg_path))


def test_central_slice_fails_with_2d(tmp_path):
    tmp_mrc_path = tmp_path / "tmp.mrc"
    data_2d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((5, 4))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_2d)
    assert not mrc_central_slice(plugin_params_central(tmp_mrc_path))


def test_central_slice_works_with_3d(tmp_path):
    tmp_mrc_path = tmp_path / "tmp.mrc"
    data_3d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((2, 2, 5))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    assert mrc_central_slice(plugin_params_central(tmp_mrc_path)) == str(
        tmp_path / "tmp_thumbnail.jpeg"
    )


def test_projection_fails_with_2d(tmp_path):
    tmp_mrc_path = tmp_path / "tmp.mrc"
    data_2d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((5, 4))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_2d)
    assert not mrc_projection(plugin_params(tmp_mrc_path, False, projection="XY"))


def test_mrc_projection_works_with_3d(tmp_path):
    tmp_mrc_path = tmp_path / "tmp.mrc"
    data_3d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((2, 2, 5))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    for pj in ("XY", "xy", "YX", "yx", "YZ", "yz", "ZY", "zy", "ZX", "zx", "XZ", "xz"):
        assert mrc_projection(plugin_params(tmp_mrc_path, False, projection=pj)) == str(
            tmp_path / f"tmp_proj{pj}.jpeg"
        )
    assert not mrc_projection(plugin_params(tmp_mrc_path, False, projection="other"))


@mock.patch("cryoemservices.services.images_plugins.ImageDraw")
def test_mrc_projection_ack_with_scalebar(mock_imagedraw, tmp_path):
    tmp_mrc_path = tmp_path / "tmp.mrc"
    data_3d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((2, 2, 5))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    assert mrc_projection(
        plugin_params(tmp_mrc_path, False, 30, projection="XY")
    ) == str(tmp_path / "tmp_projXY.jpeg")
    mock_imagedraw.Draw.assert_called()
    mock_imagedraw.Draw().line.assert_called_once_with(
        (
            (20, 2 / 2 - 2 / 3 / 2),
            (20, 2 / 2 + 2 / 3 / 2),
        ),
        fill="yellow",
        width=5,
    )
    mock_imagedraw.Draw().text.assert_called_once_with(
        (25, 1),
        "2 nm",
        anchor="lm",
        font=mock.ANY,
        fill="yellow",
    )


@mock.patch("cryoemservices.services.images_plugins.ImageDraw")
def test_mrc_projection_ack_with_thickness(mock_imagedraw, tmp_path):
    tmp_mrc_path = tmp_path / "tmp.mrc"
    data_3d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((2, 2, 5))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    assert mrc_projection(
        plugin_params(tmp_mrc_path, False, 10, projection="XY", thickness=90)
    ) == str(tmp_path / "tmp_projXY.jpeg")
    mock_imagedraw.Draw.assert_called()
    mock_imagedraw.Draw().line.assert_called_once_with(
        (
            (20, 2 / 2 - 90 / 10 / 2),
            (20, 2 / 2 + 90 / 10 / 2),
        ),
        fill="yellow",
        width=5,
    )
    mock_imagedraw.Draw().text.assert_called_once_with(
        (25, 1),
        "Thickness: 9 nm",
        anchor="lm",
        font=mock.ANY,
        fill="yellow",
    )


def test_mrc_to_apng_fails_with_2d(tmp_path):
    tmp_mrc_path = tmp_path / "tmp.mrc"
    data_2d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((5, 4))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_2d)
    assert not mrc_to_apng(plugin_params_central(tmp_mrc_path))


def test_mrc_to_apng_works_with_3d(tmp_path):
    tmp_mrc_path = tmp_path / "tmp.mrc"
    data_3d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((2, 2, 5))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    assert mrc_to_apng(plugin_params_central(tmp_mrc_path)) == str(
        tmp_path / "tmp_movie.png"
    )


@mock.patch("cryoemservices.services.images_plugins.PIL.Image")
def test_mrc_to_apng_skip_rescaling(mock_pil, tmp_path):
    tmp_mrc_path = tmp_path / "tmp.mrc"
    data_3d = np.linspace(0, 49, 50, dtype=np.int16).reshape((2, 5, 5))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    assert mrc_to_apng(plugin_params_central(tmp_mrc_path, skip_rescaling=True)) == str(
        tmp_path / "tmp_movie.png"
    )

    # Check the image creation
    assert mock_pil.fromarray.call_count == 2
    assert (
        mock_pil.fromarray.mock_calls[0][1] == (data_3d[0] * 255 / 24).astype("uint8")
    ).all()
    assert (
        mock_pil.fromarray.mock_calls[2][1]
        == ((data_3d[1] - 25) * 255 / 24).astype("uint8")
    ).all()
    mock_pil.fromarray().thumbnail.assert_called_with((512, 512))


@mock.patch("cryoemservices.services.images_plugins.PIL.Image")
def test_mrc_to_apng_jitter_edge(mock_pil, tmp_path):
    tmp_mrc_path = tmp_path / "tmp.mrc"
    data_3d = np.linspace(0, 49, 50, dtype=np.int16).reshape((2, 5, 5))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    assert mrc_to_apng(
        plugin_params_central(tmp_mrc_path, skip_rescaling=True, jitter_edge=True)
    ) == str(tmp_path / "tmp_movie.png")

    # Check the image creation
    # This time the centre should match but the edge has changed
    assert mock_pil.fromarray.call_count == 2
    assert (mock_pil.fromarray.mock_calls[0][1] != data_3d[0] * 255 / 24).any()
    assert (mock_pil.fromarray.mock_calls[2][1] != (data_3d[1] - 25) * 255 / 24).any()
    assert (mock_pil.fromarray.mock_calls[0][1] == data_3d[0] * 255 / 24)[
        2:-2, 2:-2
    ].all()
    assert (mock_pil.fromarray.mock_calls[2][1] == (data_3d[1] - 25) * 255 / 24)[
        2:-2, 2:-2
    ].all()
    mock_pil.fromarray().thumbnail.assert_called_with((512, 512))


@mock.patch("cryoemservices.services.images_plugins.PIL.Image")
def test_mrc_to_apng_rescaling(mock_pil, tmp_path):
    tmp_mrc_path = tmp_path / "tmp.mrc"
    data_3d = np.linspace(0, 49, 50, dtype=np.int16).reshape((2, 5, 5))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    assert mrc_to_apng(plugin_params_central(tmp_mrc_path)) == str(
        tmp_path / "tmp_movie.png"
    )

    mean = np.mean(data_3d[0])
    sdev = np.std(data_3d[0])
    sigma_min = mean - 3 * sdev
    sigma_max = mean + 3 * sdev
    rescaled_frame = data_3d[0]
    rescaled_frame[rescaled_frame < sigma_min] = sigma_min
    rescaled_frame[rescaled_frame > sigma_max] = sigma_max

    # Check the image creation for the first frame
    # This time the frame has been rescaled
    assert mock_pil.fromarray.call_count == 2
    assert (mock_pil.fromarray.mock_calls[0][1] == rescaled_frame * 255 / 24).any()
    mock_pil.fromarray().thumbnail.assert_called_with((512, 512))


def test_mrc_to_apng_colour_works_with_3d_mask(tmp_path):
    tmp_mrc_path1 = tmp_path / "tmp1.mrc"
    tmp_mrc_path2 = tmp_path / "tmp2.mrc"
    data_3d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((2, 2, 5))
    with mrcfile.new(tmp_mrc_path1, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    with mrcfile.new(tmp_mrc_path2, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    assert mrc_to_apng_colour(
        plugin_params_apng_colour([str(tmp_mrc_path1)], mask=str(tmp_mrc_path2))
    ) == str(tmp_path / "tmp1_movie.png")
    assert (tmp_path / "tmp1_movie.png").is_file()
    assert (tmp_path / "tmp1_thumbnail.jpeg").is_file()


def test_mrc_to_apng_colour_withs_with_3d_no_mask(tmp_path):
    tmp_mrc_path1 = tmp_path / "tmp1.mrc"
    tmp_mrc_path2 = tmp_path / "tmp2.mrc"
    data_3d = np.linspace(-1000, 1000, 80, dtype=np.int16).reshape((20, 2, 2))
    with mrcfile.new(tmp_mrc_path1, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    with mrcfile.new(tmp_mrc_path2, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    assert mrc_to_apng_colour(
        plugin_params_apng_colour([str(tmp_mrc_path1), str(tmp_mrc_path2)], mask=None)
    ) == str(tmp_path / "tmp1_movie.png")
    assert (tmp_path / "tmp1_movie.png").is_file()
    assert (tmp_path / "tmp1_thumbnail.jpeg").is_file()


def test_mrc_to_apng_colour_fail_cases(tmp_path):
    tmp_mrc_path1 = tmp_path / "tmp1.mrc"
    tmp_mrc_path2 = tmp_path / "tmp2.mrc"
    data_3d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((2, 2, 5))
    with mrcfile.new(tmp_mrc_path2, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    # File not found case
    assert not mrc_to_apng_colour(
        plugin_params_apng_colour([str(tmp_mrc_path1)], mask=str(tmp_mrc_path2))
    )
    # Not an mrc case
    tmp_mrc_path1.touch()
    assert not mrc_to_apng_colour(
        plugin_params_apng_colour([str(tmp_mrc_path1)], mask=str(tmp_mrc_path2))
    )
    # 2D data case
    tmp_mrc_path1.unlink()
    data_2d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((5, 4))
    with mrcfile.new(tmp_mrc_path1, overwrite=True) as mrc:
        mrc.set_data(data_2d)
    assert not mrc_to_apng_colour(
        plugin_params_apng_colour([str(tmp_mrc_path1)], mask=str(tmp_mrc_path2))
    )
    # Non-matching volume shapes
    data_3d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((2, 5, 2))
    with mrcfile.new(tmp_mrc_path1, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    assert not mrc_to_apng_colour(
        plugin_params_apng_colour([str(tmp_mrc_path1)], mask=str(tmp_mrc_path2))
    )


def test_picked_particles_3d_central_slice_fails_with_2d(tmp_path):
    tmp_mrc_path = tmp_path / "tmp.mrc"
    coords_file = tmp_path / "coords.cbox"
    coords_file.touch()
    data_2d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((5, 4))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_2d)
    assert not picked_particles_3d_central_slice(
        plugin_params_tomo_pick(
            tmp_mrc_path,
            coords_file,
            "picked_particles_3d_central_slice",
        )
    )


@mock.patch("cryoemservices.services.images_plugins.ImageDraw")
def test_picked_particles_3d_central_slice_works_with_3d(mock_imagedraw, tmp_path):
    tmp_mrc_path = str(tmp_path / "tmp.mrc")
    coords_file = tmp_path / "coords.cbox"
    data_3d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((2, 2, 5))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    with open(coords_file, "w") as cbox:
        cbox.write(
            "data_global\n\n_version 1\n\n"
            "data_cryolo\n\nloop_\n_CoordinateX\n_CoordinateY\n_CoordinateZ\n"
            "_EstWidth\n_EstHeight\n_NumBoxes\n"
            "40 50 0 20 30 2 \n70 80 1 40 50 2\n"
        )
    assert picked_particles_3d_central_slice(
        plugin_params_tomo_pick(
            tmp_mrc_path,
            coords_file,
            "picked_particles_3d_central_slice",
        )
    ) == str(tmp_path / "coords_thumbnail.jpeg")

    # Check that the expected picking ellipses were drawn
    mock_imagedraw.Draw.assert_called()
    assert mock_imagedraw.Draw().ellipse.call_count == 2
    mock_imagedraw.Draw().ellipse.assert_any_call(
        [
            (
                40 + 20 - np.sqrt(20**2 - 1**2) / 2,
                50 + 20 - np.sqrt(30**2 - 1**2) / 2,
            ),
            (
                40 + 20 + np.sqrt(20**2 - 1**2) / 2,
                50 + 20 + np.sqrt(30**2 - 1**2) / 2,
            ),
        ],
        width=4,
        outline="#f52407",
    )
    mock_imagedraw.Draw().ellipse.assert_any_call(
        [
            (
                70 + 20 - 20,
                80 + 20 - 25,
            ),
            (
                70 + 20 + 20,
                80 + 20 + 25,
            ),
        ],
        width=4,
        outline="#f52407",
    )


def test_picked_particles_3d_central_slice_works_without_coords(tmp_path):
    tmp_mrc_path = str(tmp_path / "tmp.mrc")
    coords_file = tmp_path / "coords.cbox"
    data_3d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((2, 2, 5))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    with open(coords_file, "w") as cbox:
        cbox.write(
            "data_global\n\n_version 1\n\n"
            "data_cryolo\n\nloop_\n_CoordinateX\n_CoordinateY\n_CoordinateZ\n"
        )
    assert picked_particles_3d_central_slice(
        plugin_params_tomo_pick(
            tmp_mrc_path,
            coords_file,
            "picked_particles_3d_central_slice",
        )
    ) == str(tmp_path / "coords_thumbnail.jpeg")


def test_picked_particles_3d_apng_fails_with_2d(tmp_path):
    tmp_mrc_path = str(tmp_path / "tmp.mrc")
    coords_file = tmp_path / "coords.cbox"
    coords_file.touch()
    data_2d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((5, 4))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_2d)
    assert not picked_particles_3d_apng(
        plugin_params_tomo_pick(
            tmp_mrc_path,
            coords_file,
            "picked_particles_3d_apng",
        )
    )


@mock.patch("cryoemservices.services.images_plugins.ImageDraw")
def test_picked_particles_3d_apng_works_with_3d(mock_imagedraw, tmp_path):
    tmp_mrc_path = str(tmp_path / "tmp.mrc")
    coords_file = tmp_path / "coords.cbox"
    data_3d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((2, 2, 5))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    with open(coords_file, "w") as cbox:
        cbox.write(
            "data_global\n\n_version 1\n\n"
            "data_cryolo\n\nloop_\n_CoordinateX\n_CoordinateY\n_CoordinateZ\n"
            "_EstWidth\n_EstHeight\n_NumBoxes\n"
            "40 50 0 20 30 2 \n70 80 1 40 50 2\n"
        )
    assert picked_particles_3d_apng(
        plugin_params_tomo_pick(tmp_mrc_path, coords_file, "picked_particles_3d_apng")
    ) == str(tmp_path / "coords_movie.png")

    # Check that the expected picking ellipses were drawn
    mock_imagedraw.Draw.assert_called()
    assert mock_imagedraw.Draw().ellipse.call_count == 4
    mock_imagedraw.Draw().ellipse.assert_any_call(
        [
            (
                40 + 20 - 10,
                50 + 20 - 15,
            ),
            (
                40 + 20 + 10,
                50 + 20 + 15,
            ),
        ],
        width=4,
        outline="#f52407",
    )
    mock_imagedraw.Draw().ellipse.assert_any_call(
        [
            (
                40 + 20 - np.sqrt(20**2 - 1**2) / 2,
                50 + 20 - np.sqrt(30**2 - 1**2) / 2,
            ),
            (
                40 + 20 + np.sqrt(20**2 - 1**2) / 2,
                50 + 20 + np.sqrt(30**2 - 1**2) / 2,
            ),
        ],
        width=4,
        outline="#f52407",
    )
    mock_imagedraw.Draw().ellipse.assert_any_call(
        [
            (
                70 + 20 - 20,
                80 + 20 - 25,
            ),
            (
                70 + 20 + 20,
                80 + 20 + 25,
            ),
        ],
        width=4,
        outline="#f52407",
    )
    mock_imagedraw.Draw().ellipse.assert_any_call(
        [
            (
                70 + 20 - np.sqrt(40**2 - 1**2) / 2,
                80 + 20 - np.sqrt(50**2 - 1**2) / 2,
            ),
            (
                70 + 20 + np.sqrt(40**2 - 1**2) / 2,
                80 + 20 + np.sqrt(50**2 - 1**2) / 2,
            ),
        ],
        width=4,
        outline="#f52407",
    )


def test_picked_particles_3d_apng_works_without_coords(tmp_path):
    tmp_mrc_path = str(tmp_path / "tmp.mrc")
    coords_file = tmp_path / "coords.cbox"
    data_3d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((2, 2, 5))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    with open(coords_file, "w") as cbox:
        cbox.write(
            "data_global\n\n_version 1\n\n"
            "data_cryolo\n\nloop_\n_CoordinateX\n_CoordinateY\n_CoordinateZ\n"
        )
    assert picked_particles_3d_apng(
        plugin_params_tomo_pick(tmp_mrc_path, coords_file, "picked_particles_3d_apng")
    ) == str(tmp_path / "coords_movie.png")


def plugin_params_tiff_to_apng(
    input_file: Path | None = None,
    output_file: Path | None = None,
    target_size: tuple[int | None, int | None] | None = None,
    color: str | None = None,
):
    def params(key):
        p = {
            "parameters": {"images_command": "tiff_to_apng"},
            "input_file": input_file,
            "output_file": output_file,
            "target_size": target_size,
            "color": color,
        }
        return p.get(key)

    return params


# Programmatically generate test matrix
tiff_to_apng_test_params = []
for frame in (1, 5):
    for is_rgb in (True, False):
        for color in (
            None,
            "gray",
            "red",
            "green",
            "blue",
            "cyan",
            "magenta",
            "yellow",
        ):
            for resize in (True, False):
                tiff_to_apng_test_params.append((frame, is_rgb, color, resize))
tiff_to_apng_test_matrix = [
    # Input file | Output file | Frames | Initial image is RGB? | Color | Resize image
    (f"test_{n}.tiff", f".thumbnails/test_{n}.png", frames, is_rgb, color, resize)
    for n, (frames, is_rgb, color, resize) in enumerate(tiff_to_apng_test_params)
]


@pytest.mark.parametrize("test_params", tiff_to_apng_test_matrix)
def test_tiff_to_apng(
    tmp_path: Path,
    test_params: tuple[str, str, int, bool, str | None, bool],
):
    # Unpack test parameters
    input_file_name, output_file_name, num_frames, is_rgb, color, resize = test_params
    target_h, target_w = (8, 32) if resize else (None, None)  # height, width

    # Construct full file paths for input and output files
    input_file = tmp_path / input_file_name
    input_file.parent.mkdir(parents=True, exist_ok=True)
    output_file = tmp_path / output_file_name

    # Create test image using 'tifffile'
    initial_h, initial_w = (16, 64)
    arr = np.linspace(0, 255, 1024).reshape((initial_h, initial_w))
    if num_frames > 1:
        arr = np.asarray([arr for f in range(num_frames)])
    arr = arr.astype("uint8")
    if is_rgb:
        arr = convert_to_rgb(arr, "gray")
    tf.imwrite(
        input_file,
        arr,
        shape=arr.shape,
        dtype=str(arr.dtype),
        imagej=True,
    )

    # Run the function and check that the outputs are as expected
    assert tiff_to_apng(
        plugin_params_tiff_to_apng(input_file, output_file, (target_h, target_w), color)
    ) == str(output_file)
    assert output_file.exists()

    # Open the output file and inspect image propoerties
    output_img = PIL.Image.open(output_file)

    # Incoming RGB images will stay RGB; if 'color' is set, 8-bit images will be converted
    assert output_img.mode == "RGB" if is_rgb or color is not None else "L"

    # Check that image has been resized as specified
    assert output_img.size == (target_w, target_h) if resize else (initial_w, initial_h)

    # Check that the number of image frames is preserved
    frame_counter = 0
    output_img.seek(0)
    try:
        while True:
            frame_counter += 1
            output_img.seek(output_img.tell() + 1)
    except EOFError:
        pass
    assert frame_counter == num_frames


@mock.patch("cryoemservices.services.images_plugins.ImageDraw")
def test_tilt_image_alignment_works_3d(mock_imagedraw, tmp_path):
    tmp_mrc_path = tmp_path / "tmp.mrc"
    aln_file = tmp_path / "tmp.aln"

    # Rotation of tomogram relative to tilts
    angle = 86.7
    rotation_matrix = [
        [np.cos((angle - 90) * np.pi / 180), -np.sin((angle - 90) * np.pi / 180)],
        [np.sin((angle - 90) * np.pi / 180), np.cos((angle - 90) * np.pi / 180)],
    ]

    # Construct mrc and aln file
    data_3d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((2, 2, 5))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_3d)
    with open(aln_file, "w") as aln:
        aln.write(
            "# Header line\n# Second header line\n"
            f"0 {angle} 1 -1.5 2.5 1 1 1 0 -3.0\n0 {angle} 1 1022.2 21.2 1 1 1 0 6.0"
        )

    # Send to image creation
    assert tilt_series_alignment(plugin_params(tmp_mrc_path, True, 2)) == str(
        tmp_path / "tmp_alignment.jpeg"
    )

    # Check that the expected picking ellipses were drawn
    mock_imagedraw.Draw.assert_called()
    assert mock_imagedraw.Draw().polygon.call_count == 2
    ellipse_calls = mock_imagedraw.Draw().polygon.call_args_list
    assert ellipse_calls[0][1] == {"width": 19, "outline": "#0af549"}
    assert ellipse_calls[1][1] == {"width": 20, "outline": "#f52407"}

    # Check the box corners are right for the first ellipse
    x_cen_a = 2.5 - 1.5 / 2
    y_cen_a = 1 + 2.5 / 2
    x_shift_1 = 2.5 * rotation_matrix[0][0]
    y_shift_a1 = rotation_matrix[0][1] * np.cos(-3 * np.pi / 180)
    x_shift_2 = 2.5 * rotation_matrix[1][0]
    y_shift_a2 = rotation_matrix[1][1] * np.cos(-3 * np.pi / 180)
    assert np.isclose(ellipse_calls[0][0][0][0][0], x_cen_a + x_shift_1 + y_shift_a1)
    assert np.isclose(ellipse_calls[0][0][0][0][1], y_cen_a + x_shift_2 + y_shift_a2)
    assert np.isclose(ellipse_calls[0][0][0][1][0], x_cen_a - x_shift_1 + y_shift_a1)
    assert np.isclose(ellipse_calls[0][0][0][1][1], y_cen_a - x_shift_2 + y_shift_a2)
    assert np.isclose(ellipse_calls[0][0][0][2][0], x_cen_a - x_shift_1 - y_shift_a1)
    assert np.isclose(ellipse_calls[0][0][0][2][1], y_cen_a - x_shift_2 - y_shift_a2)
    assert np.isclose(ellipse_calls[0][0][0][3][0], x_cen_a + x_shift_1 - y_shift_a1)
    assert np.isclose(ellipse_calls[0][0][0][3][1], y_cen_a + x_shift_2 - y_shift_a2)

    # Check the box corners are right for the second ellipse
    x_cen_b = 2.5 + 1022.2 / 2
    y_cen_b = 1 + 21.2 / 2
    y_shift_b1 = rotation_matrix[0][1] * np.cos(6 * np.pi / 180)
    y_shift_b2 = rotation_matrix[1][1] * np.cos(6 * np.pi / 180)
    assert np.isclose(ellipse_calls[1][0][0][0][0], x_cen_b + x_shift_1 + y_shift_b1)
    assert np.isclose(ellipse_calls[1][0][0][0][1], y_cen_b + x_shift_2 + y_shift_b2)
    assert np.isclose(ellipse_calls[1][0][0][1][0], x_cen_b - x_shift_1 + y_shift_b1)
    assert np.isclose(ellipse_calls[1][0][0][1][1], y_cen_b - x_shift_2 + y_shift_b2)
    assert np.isclose(ellipse_calls[1][0][0][2][0], x_cen_b - x_shift_1 - y_shift_b1)
    assert np.isclose(ellipse_calls[1][0][0][2][1], y_cen_b - x_shift_2 - y_shift_b2)
    assert np.isclose(ellipse_calls[1][0][0][3][0], x_cen_b + x_shift_1 - y_shift_b1)
    assert np.isclose(ellipse_calls[1][0][0][3][1], y_cen_b + x_shift_2 - y_shift_b2)


@mock.patch("cryoemservices.services.images_plugins.ImageDraw")
def test_tilt_image_alignment_works_2d(mock_imagedraw, tmp_path):
    tmp_mrc_path = tmp_path / "tmp.mrc"
    aln_file = tmp_path / "tmp.aln"

    # Construct mrc and aln file
    data_2d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((5, 4))
    with mrcfile.new(tmp_mrc_path, overwrite=True) as mrc:
        mrc.set_data(data_2d)
    with open(aln_file, "w") as aln:
        aln.write(
            "# Header line\n# Second header line\n"
            "0 86.7 1 -10.5 2.5 1 1 1 0 -3.0\n0 86.7 1 102.2 21.2 1 1 1 0 6.0"
        )

    # Send to image creation
    assert tilt_series_alignment(plugin_params(tmp_mrc_path, True, 2)) == str(
        tmp_path / "tmp_alignment.jpeg"
    )

    # Check that the expected picking ellipses were drawn - skip checking all the rest
    mock_imagedraw.Draw.assert_called()
    assert mock_imagedraw.Draw().polygon.call_count == 2
    ellipse_calls = mock_imagedraw.Draw().polygon.call_args_list
    assert ellipse_calls[0][1] == {"width": 19, "outline": "#f5e90a"}
    assert ellipse_calls[1][1] == {"width": 20, "outline": "#f5a927"}


def test_interfaces_without_keys():
    """Test that file path keys are required"""
    assert not mrc_to_jpeg(lambda x: None)
    assert not picked_particles(lambda x: None)
    assert not mrc_central_slice(lambda x: None)
    assert not mrc_projection(lambda x: None)
    assert not mrc_to_apng(lambda x: None)
    assert not picked_particles_3d_central_slice(lambda x: None)
    assert not picked_particles_3d_apng(lambda x: None)
    assert not tilt_series_alignment(lambda x: None)


def test_interfaces_without_files(tmp_path):
    """Assert rejections if files specified do not exist"""
    (tmp_path / "test.mrc").touch()
    (tmp_path / "coords.cbox").touch()

    assert not mrc_to_jpeg(plugin_params(tmp_path / "not.mrc", False))
    assert not picked_particles(plugin_params_parpick(tmp_path / "not.mrc", "test.jpg"))
    assert not picked_particles(plugin_params_parpick(tmp_path / "test.jpeg", None))
    assert not picked_particles(plugin_params_parpick(tmp_path / "test.mrc", None))
    assert not mrc_central_slice(plugin_params_central(tmp_path / "not.mrc"))
    assert not mrc_projection(plugin_params(tmp_path / "not.mrc", False))
    assert not mrc_to_apng(plugin_params_central(tmp_path / "not.mrc"))
    assert not picked_particles_3d_central_slice(
        plugin_params_tomo_pick(
            tmp_path / "not.mrc", tmp_path / "coords.cbox", "picked_particles_3d_apng"
        )
    )
    assert not picked_particles_3d_central_slice(
        plugin_params_tomo_pick(
            tmp_path / "test.mrc", tmp_path / "not.cbox", "picked_particles_3d_apng"
        )
    )
    assert not picked_particles_3d_apng(
        plugin_params_tomo_pick(
            tmp_path / "not.mrc", tmp_path / "coords.cbox", "picked_particles_3d_apng"
        )
    )
    assert not picked_particles_3d_apng(
        plugin_params_tomo_pick(
            tmp_path / "test.mrc", tmp_path / "not.cbox", "picked_particles_3d_apng"
        )
    )
    assert not tilt_series_alignment(plugin_params(tmp_path / "not.mrc", True, 1))
