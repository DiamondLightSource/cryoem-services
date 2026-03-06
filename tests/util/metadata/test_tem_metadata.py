import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from defusedxml.ElementTree import parse

from cryoemservices.util.metadata import cleanup_xml
from cryoemservices.util.metadata.tem import parse_atlas_metadata
from tests.test_utils.tem import create_tem_atlas_metadata

atlas_id = 12345678
pixel_size = 4e-7


@pytest.fixture
def atlas_metadata_file(tmp_path: Path):
    metadata = create_tem_atlas_metadata(
        pixel_size=pixel_size,
        use_namespaces=True,
    )
    tree = ET.ElementTree(metadata)
    ET.indent(tree, space="  ")
    save_path = tmp_path / f"Atlas_{atlas_id}.xml"
    tree.write(save_path, encoding="utf-8")
    return save_path


def test_parse_atlas_metadata(
    atlas_metadata_file: Path,
):
    metadata = parse_atlas_metadata(cleanup_xml(parse(atlas_metadata_file).getroot()))
    assert metadata["x_pixel_size"] == pixel_size
    assert metadata["y_pixel_size"] == pixel_size
