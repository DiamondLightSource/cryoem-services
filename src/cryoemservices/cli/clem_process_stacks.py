"""
A cryoEM service command line interface to facilitate the processing of image stacks
from a single CLEM dataset, generating colorised composite images or image stacks for
use in the subsequent stage of the CLEM workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cryoemservices.services.clem_processing import merge_image_stacks


def run():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description=(
            "Takes image stacks of the colour channels from a CLEM dataset, processes "
            "them, and creates a colourised composite image or image stack for that "
            "dataset."
        )
    )
    # List of paths of files to merge (mandatory)
    parser.add_argument(
        dest="images",
        type=str,
        nargs="+",  # Gather as list by default and raise warning if not provided
        help="Full file paths to the image stacks to be processed",
    )
    # Metadata file (optional)
    parser.add_argument(
        "--metadata",
        default=None,
        type=str,
        help="Full file path to the metadata file associated with this dataset",
    )
    # Determine how the image is flattened
    parser.add_argument(
        "--flatten",
        default="mean",
        type=str,
        help="Choose if the image stacks should be flattened, and how to do so",
    )
    # Determine what image registration protocol to implement
    parser.add_argument(
        "--registration",
        default=None,
        type=str,
        help="Choose the type of image registration protocol to use",
    )
    # Parse the arguments
    args = parser.parse_args()

    # Resolve the arguments internally to pass on to the function
    # Resolve image stacks parameter
    if any((not isinstance(file, str) for file in args.images)):
        raise TypeError(
            "One or more of the image files provided are of an invalid type"
        )

    image_files = [Path(file) for file in args.images if Path(file).exists()]
    if len(image_files) == 0:
        raise FileNotFoundError("No valid file paths provided")

    # Resolve metadata parameter
    if isinstance(args.metadata, str):
        if Path(args.metadata).exists():
            metadata_file = Path(args.metadata)
        else:
            print(
                "The metadata file provided doesn't exist; searching for the metadata "
                "file using the default settings"
            )
            metadata_file = None
    elif args.metadata is None:
        metadata_file = args.metadata
    else:
        raise TypeError("The metadata parameter is of an invalid type")

    # Resolve image flattening parameter
    if isinstance(args.flatten, str):
        flatten = None if args.flatten == "null" else args.flatten
    else:
        raise TypeError("The flatten parameter is of an invalid type")

    # Resolve image registration parameter
    if isinstance(args.registration, str):
        registration = None if args.registration == "null" else args.flatten
    else:
        raise TypeError("The registration parameter is of an invalid type")

    # Run function
    composite_image = merge_image_stacks(
        image_files=image_files,
        metadata_file=metadata_file,
        flatten=flatten,
        registration=registration,
    )
    if composite_image:
        return True
