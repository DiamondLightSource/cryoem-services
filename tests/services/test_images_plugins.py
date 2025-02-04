from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, NamedTuple
from unittest import mock

import mrcfile
import numpy as np
import workflows

from cryoemservices.services.images import PluginInterface
from cryoemservices.services.images_plugins import (
    mrc_central_slice,
    mrc_to_apng,
    mrc_to_jpeg,
    picked_particles,
    picked_particles_3d_apng,
    picked_particles_3d_central_slice,
)


class FunctionParameter(NamedTuple):
    rw: workflows.recipe.wrapper.RecipeWrapper
    parameters: Callable
    message: dict[str, Any]


def plugin_params(jpeg_path: Path, all_frames: bool):
    def params(key):
        p = {
            "parameters": {"images_command": "mrc_to_jpeg"},
            "file": jpeg_path.with_suffix(".mrc"),
            "all_frames": all_frames,
        }
        return p.get(key)

    return FunctionParameter(rw=None, parameters=params, message={})


def plugin_params_central(jpeg_path):
    def params(key):
        p = {
            "parameters": {"images_command": "mrc_central_slice"},
            "file": jpeg_path.with_suffix(".mrc"),
        }
        return p.get(key)

    return FunctionParameter(rw=None, parameters=params, message={})


def plugin_params_parpick(jpeg_path, outfile):
    def params(key, default=None):
        p = {
            "parameters": {"images_command": "picked_particles"},
            "file": jpeg_path,
            "coordinates": [("0", "1"), ("2", "2")],
            "angpix": 0.5,
            "diameter": 190,
            "outfile": outfile,
        }
        return p.get(key) or default

    return FunctionParameter(rw=None, parameters=params, message={})


def plugin_params_tomo_pick(input_file, coords_file, command):
    def params(key):
        p = {
            "parameters": {"images_command": command},
            "file": input_file,
            "coordinates_file": coords_file,
            "box_size": 40,
        }
        return p.get(key)

    return FunctionParameter(rw=None, parameters=params, message={})


def test_contract_with_images_service():
    # Check that we do not declare any keys that are unknown in the images service
    assert set(FunctionParameter._fields).issubset(PluginInterface._fields)

    for key, annotation in FunctionParameter.__annotations__.items():
        if annotation is Any:
            continue
        upstream_type = PluginInterface.__annotations__[key]
        if annotation == upstream_type:
            continue
        if not hasattr(annotation, "_name") or not hasattr(upstream_type, "_name"):
            raise TypeError(
                f"Parameter {key!r} with local type {annotation!r} does not match upstream type {upstream_type!r}"
            )
        assert annotation._name == upstream_type._name


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
