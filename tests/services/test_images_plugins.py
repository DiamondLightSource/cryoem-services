from __future__ import annotations

import pathlib
from typing import Any, Callable, NamedTuple

import mrcfile
import numpy as np
import workflows

from cryoemservices.services.images import PluginInterface
from cryoemservices.services.images_plugins import (
    mrc_central_slice,
    mrc_to_apng,
    mrc_to_jpeg,
    picked_particles,
)


class FunctionParameter(NamedTuple):
    rw: workflows.recipe.wrapper.RecipeWrapper
    parameters: Callable
    message: dict[str, Any]


def plugin_params(jpeg_path):
    def params(key):
        p = {
            "parameters": {"images_command": "mrc_to_jpeg"},
            "file": jpeg_path.with_suffix(".mrc"),
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
    assert not mrc_to_jpeg(plugin_params(jpeg_path))


def test_mrc_to_jpeg_ack_when_file_exists(tmp_path):
    jpeg_path = tmp_path / "convert_test.jpeg"
    test_data = np.arange(9, dtype=np.int8).reshape(3, 3)
    with mrcfile.new(jpeg_path.with_suffix(".mrc")) as mrc:
        mrc.set_data(test_data)

    assert mrc_to_jpeg(plugin_params(jpeg_path)) == jpeg_path
    assert jpeg_path.is_file()


def test_picked_particles_processes_when_basefile_exists(tmp_path):
    base_mrc_path = str(tmp_path / "base.mrc")
    out_jpeg_path = str(tmp_path / "processed.jpeg")
    test_data = np.arange(16, dtype=np.int8).reshape(4, 4)
    with mrcfile.new(base_mrc_path) as mrc:
        mrc.set_data(test_data)

    assert (
        picked_particles(plugin_params_parpick(base_mrc_path, out_jpeg_path))
        == out_jpeg_path
    )


def test_picked_particles_fails_when_basefile_does_not_exist(tmp_path):
    base_mrc_path = str(tmp_path / "base.mrc")
    out_jpeg_path = str(tmp_path / "processed.jpeg")
    assert not picked_particles(plugin_params_parpick(base_mrc_path, out_jpeg_path))


def test_central_slice_fails_with_2d(tmp_path):
    tmp_mrc_path = str(tmp_path / "tmp.mrc")
    mrc = mrcfile.new(tmp_mrc_path, overwrite=True)
    data_2d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((5, 4))
    mrc.set_data(data_2d)
    mrc.close()
    assert not mrc_central_slice(plugin_params_central(pathlib.Path(tmp_mrc_path)))


def test_central_slice_works_with_3d(tmp_path):
    tmp_mrc_path = str(tmp_path / "tmp.mrc")
    mrc = mrcfile.new(tmp_mrc_path, overwrite=True)
    data_3d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((2, 2, 5))
    mrc.set_data(data_3d)
    mrc.close()
    assert mrc_central_slice(plugin_params_central(pathlib.Path(tmp_mrc_path))) == str(
        tmp_path / "tmp_thumbnail.jpeg"
    )


def test_mrc_to_apng_fails_with_2d(tmp_path):
    tmp_mrc_path = str(tmp_path / "tmp.mrc")
    mrc = mrcfile.new(tmp_mrc_path, overwrite=True)
    data_2d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((5, 4))
    mrc.set_data(data_2d)
    mrc.close()
    assert not mrc_to_apng(plugin_params_central(pathlib.Path(tmp_mrc_path)))


def test_mrc_to_apng_works_with_3d(tmp_path):
    tmp_mrc_path = str(tmp_path / "tmp.mrc")
    mrc = mrcfile.new(tmp_mrc_path, overwrite=True)
    data_3d = np.linspace(-1000, 1000, 20, dtype=np.int16).reshape((2, 2, 5))
    mrc.set_data(data_3d)
    mrc.close()
    assert mrc_to_apng(plugin_params_central(pathlib.Path(tmp_mrc_path))) == str(
        tmp_path / "tmp_movie.png"
    )
