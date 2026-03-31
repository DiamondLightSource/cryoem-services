import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast

import pytest
from defusedxml.ElementTree import parse

from cryoemservices.util.metadata.fib import find_image_elements, parse_fib_metadata
from tests.test_utils.fib import create_fib_xml_metadata

# Test values to use for the FIB metadata
test_datasets = [
    {
        "name": name,
        "relative_path": relative_path,
        "center_x": cx,
        "center_y": cy,
        "center_z": cz,
        "size_x": sx,
        "size_y": sy,
        "size_z": sz,
        "rotation_angle": ra,
    }
    for (name, relative_path, cx, cy, cz, sx, sy, sz, ra) in (
        (
            "Electron Snapshot",
            "LayersData/Layer/Electron Snapshot",
            -0.002,
            -0.004,
            0.00000008,
            0.0036,
            0.0024,
            0.0,
            3.1415926535897931,
        ),
        (
            "Electron Snapshot (2)",
            "LayersData/Layer/Electron Snapshot (2)",
            -0.002,
            -0.004,
            0.00000008,
            0.0036,
            0.0024,
            0.0,
            3.1415926535897931,
        ),
        (
            "Electron Snapshot (3)",
            "LayersData/Layer/Electron Snapshot (3)",
            0.002,
            0.004,
            0.00000008,
            0.0036,
            0.0024,
            0.0,
            3.1415926535897931,
        ),
        (
            "Electron Snapshot (4)",
            "LayersData/Layer/Electron Snapshot (4)",
            0.002,
            0.004,
            0.00000008,
            0.0036,
            0.0024,
            0.0,
            3.1415926535897931,
        ),
    )
]


@pytest.fixture
def fib_metadata_file(tmp_path: Path):
    metadata = create_fib_xml_metadata(
        "test-project",
        test_datasets,
    )
    tree = ET.ElementTree(metadata)
    ET.indent(tree, space="  ")
    save_path = tmp_path / "EMproject.emxml"
    tree.write(save_path, encoding="utf-8")
    return save_path


def test_find_image_elements(fib_metadata_file: Path):
    root = parse(fib_metadata_file).getroot()
    assert len(find_image_elements(root)) == len(test_datasets)


@pytest.mark.parametrize("dataset_num", list(range(len(test_datasets))))
def test_parse_fib_metadata(
    fib_metadata_file: Path,
    dataset_num: int,
):
    data_to_compare = test_datasets[dataset_num]
    fib_metadata = parse_fib_metadata(
        find_image_elements(parse(fib_metadata_file).getroot())[dataset_num]
    )

    x_len = cast(float, data_to_compare["size_x"])
    y_len = cast(float, data_to_compare["size_y"])
    cx = cast(float, data_to_compare["center_x"])
    cy = cast(float, data_to_compare["center_y"])
    extent = [
        cx - (x_len / 2),
        cx + (x_len / 2),
        cy - (y_len / 2),
        cy + (y_len / 2),
    ]

    assert fib_metadata["name"] == data_to_compare["name"]
    assert fib_metadata["relative_file_path"] == data_to_compare["relative_path"]
    assert fib_metadata["x_len"] == x_len
    assert fib_metadata["y_len"] == y_len
    assert fib_metadata["x_center"] == cx
    assert fib_metadata["y_center"] == cy
    assert fib_metadata["extent"] == extent
    assert fib_metadata["rotation_angle"] == data_to_compare["rotation_angle"]
