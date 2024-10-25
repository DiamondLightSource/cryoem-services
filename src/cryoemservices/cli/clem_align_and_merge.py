"""
A cryoEM service command line interface to facilitate the processing of image stacks
from a single CLEM dataset, generating colorised composite images or image stacks for
use in the subsequent stage of the CLEM workflow.
"""

from __future__ import annotations

import argparse
from ast import literal_eval
from pathlib import Path

from cryoemservices.services.clem_align_and_merge import align_and_merge_stacks


def run():
    """
    Create argument parser and add arguments
    """
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
        help=(
            "Full file paths to the image stacks to be processed in the format "
            "'path/to/file_1' 'path/to/file_2' ... '/path/to/file_n'"
        ),
    )
    # Metadata file (optional)
    parser.add_argument(
        "--metadata",
        default=None,
        type=str,
        help=(
            "Full file path to the metadata file associated with this dataset. If no "
            "metadata file is provided, the function will attempt to find the matching "
            "metadata file."
        ),
    )
    # Align image stacks before flattening
    parser.add_argument(
        "--pre_align_stack",
        default=None,
        type=str,
        help=(
            "NOT IMPLEMENTED YET. \n"
            "Choose whether to align the image stacks before flattening, and how. \n"
            "DEFAULT: 'null'"
        ),
    )
    # Determine how the image is flattened
    parser.add_argument(
        "--flatten",
        default="mean",
        type=str,
        help=(
            "Choose whether to flatten the image stacks, and how. \n"
            "DEFAULT: 'mean' \n"
            "VALUES: ['null', 'min', and 'max']"
        ),
    )
    # Determine what image registration protocol to implement
    parser.add_argument(
        "--align_stacks",
        default=None,
        type=str,
        help=(
            "NOT IMPLEMENTED YET. \n"
            "Choose whether to align the image stacks before merging, and how. \n"
            "DEFAULT: 'null'"
        ),
    )
    # Add a debug statement
    parser.add_argument(
        "--debug",
        action="store_true",
        help=("Print additional messages to check functions work as intended"),
    )
    # Parse the arguments
    args = parser.parse_args()

    """
    Parse arguments
    """
    # Resolve image stacks parameter
    # Validate input
    images_arg: list[str] = args.images
    if any((not isinstance(file, str) for file in args.images)):
        raise TypeError(
            "One or more of the image files provided are of an invalid type"
        )
    # Check if a stringified list has been provided (for submission in recipes)
    if (
        len(images_arg) == 1
        and images_arg[0].startswith(("['", '["'))
        and images_arg[0].endswith(("']", '"]'))
    ):
        try:
            image_files: list[Path] = [
                Path(file)
                for file in literal_eval(images_arg[0])
                if Path(file).exists()
            ]
        except Exception:
            raise Exception("Unable to evaluate stringified list")
    # Evaluate normally as list of strings otherwise
    else:
        image_files = [Path(file) for file in images_arg if Path(file).exists()]
    # Check that files have been found correctly
    if len(image_files) == 0:
        raise FileNotFoundError("No valid file paths provided")

    # File list debugging
    print("Found all provided image files")
    if args.debug:
        [print(file) for file in image_files]

    # Resolve metadata parameter
    if isinstance(args.metadata, str):
        # Accept "null" as a keyword for the metadata argument
        if args.metadata == "null":
            metadata_file = None
        elif Path(args.metadata).exists():
            metadata_file = Path(args.metadata)
        else:
            print(
                "The metadata file provided doesn't exist; will use the metadata file "
                "found using the default settings"
            )
            metadata_file = None
    elif args.metadata is None:
        metadata_file = args.metadata
    else:
        raise TypeError("The metadata parameter is of an invalid type")

    # Resolve pre-flattening alignment parameter
    if isinstance(args.pre_align_stack, str) or args.pre_align_stack is None:
        # Use "null" as a stand-in for None
        pre_align_stack = (
            None if args.pre_align_stack == "null" else args.pre_align_stack
        )
    else:
        raise TypeError("Invalid type for pre-alignment parameter")

    # Resolve image flattening parameter
    if isinstance(args.flatten, str) or args.flatten is None:
        # Use "null" as a stand-in for None
        flatten = None if args.flatten == "null" else args.flatten
    else:
        raise TypeError("Invalid type for flattening parameter")

    # Resolve image registration parameter
    if isinstance(args.align_stacks, str) or args.align_stacks is None:
        # Use "null" as a stand-in for None
        align_stacks = None if args.align_stacks == "null" else args.align_stacks
    else:
        raise TypeError("Invalid type for image alignment parameter")

    """
    Run function
    """
    composite_image = align_and_merge_stacks(
        image_files=image_files,
        metadata_file=metadata_file,
        pre_align_stack=pre_align_stack,
        flatten=flatten,
        align_stacks=align_stacks,
        print_messages=True,  # Print messages when used as a CLI
        debug=True,  # Print debug messages
    )

    if composite_image is not None:
        print("Done")
