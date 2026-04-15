import xml.etree.ElementTree as ET
from typing import Any


def create_dataset_element(
    id: int,
    name: str,
    relative_path: str,
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float,
    size_y: float,
    size_z: float,
    rotation_angle: float,
):
    # Create dataset node
    dataset = ET.Element("Dataset")
    # ID node
    id_node = ET.Element("Id")
    id_node.text = str(id)
    dataset.append(id_node)

    # Name node
    name_node = ET.Element("Name")
    name_node.text = name
    dataset.append(name_node)

    # Stage position node
    box_center = ET.Element("BoxCenter")
    for tag, value in (
        ("CenterX", center_x),
        ("CenterY", center_y),
        ("CenterZ", center_z),
    ):
        node = ET.Element(tag)
        node.text = str(value)
        box_center.append(node)
    dataset.append(box_center)

    # Image size node
    box_size = ET.Element("BoxSize")
    for tag, value in (
        ("SizeX", size_x),
        ("SizeY", size_y),
        ("SizeZ", size_z),
    ):
        node = ET.Element(tag)
        node.text = str(value)
        box_size.append(node)
    dataset.append(box_size)

    # Rotation angle
    angle_node = ET.Element("RotationAngle")
    angle_node.text = str(rotation_angle)
    dataset.append(angle_node)

    # Relative path
    image_path_node = ET.Element("FinalImages")
    image_path_node.text = relative_path.replace("/", "\\")
    dataset.append(image_path_node)

    return dataset


def create_fib_xml_metadata(
    project_name: str,
    datasets: list[dict[str, Any]],
):
    # Create root node
    root = ET.Element("EMProject")

    # Project name node
    project_name_node = ET.Element("ProjectName")
    project_name_node.text = project_name
    root.append(project_name_node)

    # Datasets node
    datasets_node = ET.Element("Datasets")
    for id, dataset in enumerate(datasets):
        datasets_node.append(create_dataset_element(id, **dataset))
    root.append(datasets_node)

    return root
