"""
Contains functions needed in order to run a service to merge image stacks into
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

# import numpy as np

# from cryoemservices.clem.images import (
#     convert_to_rgb,
#     create_composite_image,
#     flatten_image,
# )
# from cryoemservices.clem.xml import get_image_elements


def merge_image_stacks(
    files: Union[Path, list[Path]],
    metadata: Optional[Path] = None,
    flatten: bool = True,
):
    """
    A cryoEM service (eventually) to create composite images from component image
    stacks in the series.

    This will (eventually) take a message containing the image stack and metadata
    files associated with an image series, along with settings for how the stacks
    should be processed (a flattened 2D image or a 3D image stack).
    """

    # Validate inputs

    # Turn single entry into a list
    if isinstance(files, Path):
        files = [files]
    # Check that a value has been provided
    if len(files) == 0:
        raise ValueError("No image stack file paths have been provided")

    # Check that files have the same parent directories
    if len({file.parents[1] for file in files}) > 1:
        raise Exception(
            "The files provided come from different directories, and might not "
            "be part of the same series"
        )
    # Get parent directory
    parent_dir = list({file.parents[1] for file in files})[0]
    # Validate parent directory
    ## TO DO

    # Find metadata file if none was provided
    if metadata is None:
        if len(list((parent_dir / "metadata").glob("*.xml"))) == 0:
            raise FileNotFoundError(
                "No metadata file was found at the default directory"
            )
        if len(list((parent_dir / "metadata").glob("*.xml"))) > 1:
            raise ValueError(
                "More than one metadata file was found at the default directory"
            )
        metadata = list((parent_dir / "metadata").glob("*.xml"))[0]

    # Load image stacks
    colors: list[str] = []
    # stacks: list[np.ndarray] = []

    for file in files:
        colors.append(
            file.stem
        )  # If I use this logic, only the raw image stacks can be stored in this folder

        # Load file as array
        ## TO DO
    # Flatten images
    if flatten is True:
        ## TO DO
        pass

    # Colorise images

    return True
