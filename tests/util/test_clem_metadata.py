from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

import pytest

from cryoemservices.util.clem_metadata import (
    find_image_elements,
    get_channel_info,
    get_dimension_info,
    get_tile_scan_info,
)
from tests.test_utils.clem import add_dataset_element, create_lif_xml_metadata

visit_id = "test_visit"
root_folder = "images"
lif_file_name = "test_lif.lif"


@pytest.fixture
def lif_xml_file(tmp_path: Path):
    # Construct file path and create necessary directories
    lif_metadata_dir = tmp_path / visit_id / root_folder / "metadata"
    lif_metadata_dir.mkdir(parents=True, exist_ok=True)

    # Set up the information to be used
    datasets = [
        {
            # Create an overview dataset
            "dataset_type": "overview",
            "dataset_name": "Overview 1",
            "datasets": [
                {
                    "series_name": "Image 1",
                    "colors": ["gray", "green", "red"],
                    "bit_depth": 16,
                    "x_pix": 1024,
                    "y_pix": 1024,
                    "pixel_size": 0.00000025,
                    "num_frames": 1,
                    "z_size": 0.00000040,
                    "num_tiles": 40,
                    "tile_offset": 0.00025,
                    "collection_mode": "grid",
                },
            ],
        },
        {
            # Create series datasets
            "dataset_type": "series",
            "dataset_name": "",
            "datasets": [
                {
                    "series_name": "Series001",
                    "colors": ["gray", "green", "red"],
                    "bit_depth": 16,
                    "x_pix": 2048,
                    "y_pix": 2048,
                    "pixel_size": 0.000000125,
                    "num_frames": 100,
                    "z_size": 0.00000040,
                    "num_tiles": 0,
                    "tile_offset": 0.00025,
                    "collection_mode": "",
                },
                {
                    "series_name": "Series002",
                    "colors": ["gray"],
                    "bit_depth": 16,
                    "x_pix": 2048,
                    "y_pix": 2048,
                    "pixel_size": 0.000000125,
                    "num_frames": 1,
                    "z_size": 0.00000040,
                    "num_tiles": 0,
                    "tile_offset": 0.00025,
                    "collection_mode": "",
                },
            ],
        },
        {
            # Create tile scan datasets
            "dataset_type": "tile_scan",
            "dataset_name": "TileScan 1",
            "datasets": [
                {
                    "series_name": "Position 1",
                    "colors": ["gray", "green", "red"],
                    "bit_depth": 16,
                    "x_pix": 2048,
                    "y_pix": 2048,
                    "pixel_size": 0.000000125,
                    "num_frames": 100,
                    "z_size": 0.00000040,
                    "num_tiles": 0,
                    "tile_offset": 0.00025,
                    "collection_mode": "",
                },
                {
                    "series_name": "Position 2",
                    "colors": ["gray", "green", "red"],
                    "bit_depth": 16,
                    "x_pix": 2048,
                    "y_pix": 2048,
                    "pixel_size": 0.000000125,
                    "num_frames": 100,
                    "z_size": 0.00000040,
                    "num_tiles": 0,
                    "tile_offset": 0.00025,
                    "collection_mode": "",
                },
            ],
        },
    ]

    # Run function to create LIF metadata file
    xml_metadata = create_lif_xml_metadata(
        lif_file_path=lif_metadata_dir.parent / lif_file_name,
        datasets=datasets,
    )

    # Save it as an XML file
    save_name = lif_metadata_dir / f"{Path(lif_file_name).stem.replace(' ', '_')}.xml"
    metadata_tree = ET.ElementTree(xml_metadata)
    ET.indent(metadata_tree, "  ")
    metadata_tree.write(save_name, encoding="utf-8")

    return save_name


def test_find_image_elements(
    lif_xml_file: Path,
):
    xml_metadata = ET.parse(lif_xml_file).getroot()
    metadata_dict = find_image_elements(xml_metadata)

    # Check that a dict has been returned
    assert isinstance(metadata_dict, dict)
    for value in metadata_dict.values():
        # Check that each entry is an Element
        assert isinstance(value, ET.Element)
        # Check that each entry has "./Data/Image" as a child
        assert value.find("./Data/Image")


# Construct reusable test elements for subsequent tests
test_element_labels = (
    "series_name",
    "colors",
    "bit_depth",
    "x_pix",
    "y_pix",
    "pixel_size",
    "num_frames",
    "z_size",
    "num_tiles",
    "tile_offset",
    "collection_mode",
)
test_element_values = (
    # 2D image
    (
        "Position 1",
        ["gray", "green", "red"],
        16,
        2048,
        2048,
        0.000000125,
        1,
        0.0000004,
        1,
        0.000025,
        "",
    ),
    # 2D overview (grayscale)
    (
        "Image 1",
        ["gray"],
        16,
        1024,
        1024,
        0.00000025,
        1,
        0.0000004,
        40,
        0.000025,
        "linear",
    ),
    # 2D overview
    (
        "Image 1",
        ["gray", "blue", "yellow"],
        16,
        1024,
        1024,
        0.00000025,
        1,
        0.0000004,
        50,
        0.000025,
        "grid",
    ),
    # 3D image
    (
        "Series 1",
        ["gray", "green", "red"],
        16,
        2048,
        2048,
        0.000000125,
        10,
        0.0000004,
        0,
        0.000025,
        "",
    ),
    # 3D overview (grayscale)
    (
        "Image 1",
        ["gray"],
        16,
        1024,
        1024,
        0.00000025,
        10,
        0.0000004,
        40,
        0.000025,
        "linear",
    ),
    # 3D overview
    (
        "Image 1",
        ["gray", "green", "red"],
        16,
        1024,
        1024,
        0.00000025,
        10,
        0.0000004,
        50,
        0.000025,
        "grid",
    ),
)
test_element_kwargs = [
    # Cast dictionary to dict[str, Any] to satisfy MyPy
    cast(
        dict[str, Any],
        {test_element_labels[v]: value for v, value in enumerate(values)},
    )
    for values in test_element_values
]
test_elements = [add_dataset_element(**kwargs) for kwargs in test_element_kwargs]


@pytest.mark.parametrize("test_params", enumerate(test_elements))
def test_get_channel_info(test_params: tuple[int, ET.Element]):
    # Unpack test ID and relevant element and load relevant params
    (i, test_element) = test_params
    test_element_params = test_element_kwargs[i]

    # Run function
    channel_info = get_channel_info(test_element)

    # Compare keys in extracted dict against expected values
    assert set(channel_info.keys()) == set(test_element_params["colors"])
    bit_depth: int = test_element_params["bit_depth"]
    for info in channel_info.values():
        assert info["bit_depth"] == bit_depth
        assert info["min"] == float(0)
        assert info["max"] == float(2**bit_depth)


@pytest.mark.parametrize("test_params", enumerate(test_elements))
def test_get_dimension_info(test_params: tuple[int, ET.Element]):
    # Unpack test ID and relevant element and load relevant params
    (i, test_element) = test_params
    test_element_params = test_element_kwargs[i]

    # Run function
    dimension_info = get_dimension_info(test_element)

    # Compare keys in extracted dict against expected values
    x_pix = test_element_params["x_pix"]
    y_pix = test_element_params["y_pix"]
    num_frames = test_element_params["num_frames"]
    num_tiles = test_element_params["num_tiles"]

    # Package data for x, y, and z for iterative analysis
    xyz_data = [
        (x_pix, test_element_params["pixel_size"]),
        (y_pix, test_element_params["pixel_size"]),
        (num_frames, test_element_params["z_size"]),
    ]
    # Iteratively assert contents of x, y, and z dimensions
    for d, dim_name in enumerate(("x", "y", "z")):
        if (num_frames > 1 and dim_name == "z") or dim_name in ("x", "y"):
            # Check that the number of elements are recorded correctly
            num_elements, pixel_size = xyz_data[d]
            if dim_name in ("x", "y"):
                assert dimension_info[dim_name]["num_pixels"] == num_elements
            if dim_name == "z":
                assert dimension_info[dim_name]["num_frames"] == num_elements

            # Python's floats aren't exact, so compare to within tolerances
            end_to_end_length = float(num_elements * pixel_size)
            assert math.isclose(dimension_info[dim_name]["length"], end_to_end_length)
            assert math.isclose(
                dimension_info[dim_name]["resolution"],
                (num_elements / end_to_end_length),
            )
            assert math.isclose(dimension_info[dim_name]["pixel_size"], pixel_size)
            assert dimension_info[dim_name]["units"] == "m"
        # If there are <= 1 number of frames, the z dimension should be empty
        if dim_name == "z" and num_frames <= 1:
            assert dimension_info[dim_name] == {}

    # If there are <= 1 number of tiles, the m dimensino should be empty
    if num_tiles <= 1:
        assert dimension_info["m"] == {}
    else:
        assert dimension_info["m"]["num_tiles"] == num_tiles


@pytest.mark.parametrize("test_params", enumerate(test_elements))
def test_get_tile_scan_info(test_params: tuple[int, ET.Element]):
    # Unpack test element and get corresponding parameters
    i, test_element = test_params
    test_element_params = test_element_kwargs[i]

    # Run the function
    tile_scan_info = get_tile_scan_info(test_element)

    num_tiles = test_element_params["num_tiles"]
    collection_mode = test_element_params["collection_mode"]
    if num_tiles == 0:
        num_tiles = 1
    tile_offset = test_element_params["tile_offset"]

    num_cols = math.ceil(math.sqrt(num_tiles))
    for t in range(num_tiles):
        col = t % num_cols
        row = t // num_cols
        field_x = col if collection_mode == "grid" else t
        field_y = row if collection_mode == "grid" else 0
        pos_x = float(tile_offset * (col + 1))
        pos_y = float(tile_offset * (row + 1))

        assert tile_scan_info[t]["field_x"] == field_x
        assert tile_scan_info[t]["field_y"] == field_y
        assert math.isclose(tile_scan_info[t]["pos_x"], pos_x)
        assert math.isclose(tile_scan_info[t]["pos_y"], pos_y)
