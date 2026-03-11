import xml.etree.ElementTree as ET


def find_image_elements(node: ET.Element):
    """
    Parses through the FIB XML metadata model to look for nodes containing
    image metadata information.

    For the Aquilos, the images are named "Electron Snapshot" by default,
    and subsequent images in the dataset have auto-incrementing numbers
    appended to them in the form '(N)'.
    """
    return [
        dataset
        for dataset in node.findall(".//Datasets/Dataset")
        if (name_node := dataset.find("Name")) is not None
        and name_node.text is not None
        and name_node.text.startswith("Electron Snapshot")
    ]


def parse_fib_metadata(node: ET.Element):
    """
    Takes a an XML Element containing image metadata for a single dataset,
    extracts useful values about the image, and returns it as a dictionary.
    """

    # Extract strings
    name, relative_file_path = [
        field.text
        if (field := node.find(path)) is not None and field.text is not None
        else ""
        for path in (".//Name", ".//FinalImages")
    ]
    relative_file_path = relative_file_path.replace("\\", "/")  # Windows -> POSIX path

    # Extract floats
    cx, cy, x_len, y_len, rotation_angle = [
        float(field.text)
        if (field := node.find(path)) is not None and field.text is not None
        else None
        for path in (
            ".//BoxCenter/CenterX",
            ".//BoxCenter/CenterY",
            ".//BoxSize/SizeX",
            ".//BoxSize/SizeY",
            ".//RotationAngle",
        )
    ]

    # Calculate the extent of the image on the stage
    extent = (
        [
            cx - (x_len / 2),
            cx + (x_len / 2),
            cy - (y_len / 2),
            cy + (y_len / 2),
        ]
        if cx is not None and cy is not None and x_len is not None and y_len is not None
        else [0.0, 0.0, 0.0, 0.0]
    )

    # Update dictionary with these values
    return {
        "name": name,
        "relative_file_path": relative_file_path,  # Relative to XML file parent path
        "x_len": x_len,
        "y_len": y_len,
        "units": "m",
        "x_center": cx,
        "y_center": cy,
        "extent": extent,
        "rotation_angle": rotation_angle,  # In radians
    }
