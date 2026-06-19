import xml.etree.ElementTree as ET


def parse_atlas_metadata(node: ET.Element):
    """
    Extract metadata from
    """
    x_pixel_size, y_pixel_size = [
        float(field.text)
        if (field := node.find(path)) is not None and field.text is not None
        else None
        for path in (
            ".//pixelSize/x/numericValue",
            ".//pixelSize/y/numericValue",
        )
    ]

    return {
        "x_pixel_size": x_pixel_size,
        "y_pixel_size": y_pixel_size,
    }
