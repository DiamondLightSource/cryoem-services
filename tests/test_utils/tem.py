import xml.etree.ElementTree as ET


def create_tem_atlas_metadata(
    pixel_size: float,
    use_namespaces: bool = False,
):
    schema = "http://schemas.datacontract.org/schema/1"
    ns = f"{{{schema}}}" if use_namespaces else ""

    # Root node
    root = ET.Element(f"{ns}MicrosocopeImage")

    # Spatial scale node
    spatial_scale_node = ET.Element(f"{ns}SpatialScale")
    # Pixel size node
    pixel_size_node = ET.Element(f"{ns}pixelSize")
    # Nested elements for each dimension
    for dim in ("x", "y"):
        dim_node = ET.Element(f"{ns}{dim}")
        value_node = ET.Element(f"{ns}numericValue")
        value_node.text = str(pixel_size)
        unit_node = ET.Element(f"{ns}unit")
        prefix_node = ET.Element(f"{ns}_x003C_PrefixExponent_x003E_k__BackingField")
        prefix_node.text = "1"
        symbol_node = ET.Element(f"{ns}_x003C_Symbol_x003E_k__BackingField")
        symbol_node.text = "m"
        unit_node.append(prefix_node)
        unit_node.append(symbol_node)
        for node in (value_node, unit_node):
            dim_node.append(node)
        pixel_size_node.append(dim_node)
    spatial_scale_node.append(pixel_size_node)
    root.append(spatial_scale_node)

    return root
