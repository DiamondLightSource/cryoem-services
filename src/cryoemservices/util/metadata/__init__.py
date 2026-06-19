"""
Helper functions for parsing metadata files that are not workflow-specific
"""

import xml.etree.ElementTree as ET


def cleanup_xml(node: ET.Element):
    """
    Certain XML models make use of universal namespaces instead of local ones in the
    element and attribute names. Thistakes the form of a URI encased in curly brackets
    ('{}'), followed immediately by the local name of the element. This notation makes
    reading and parsing the XML data model less straightforward.

    This function creates a new copy of the XML Element with all the namespace string
    patterns fully removed.
    """

    def _strip_namespace(field: str):
        return field.split("}", 1)[1] if field.startswith("{") else field

    # Create a new copy of the current node instead of editing in-place
    new = ET.Element(
        _strip_namespace(node.tag),
        {_strip_namespace(key): value for key, value in node.attrib.items()},
    )
    new.text = node.text
    new.tail = node.tail

    for child in node:
        new.append(cleanup_xml(child))
    return new
