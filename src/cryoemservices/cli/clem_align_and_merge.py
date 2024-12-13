"""
A cryoEM service command line interface to facilitate the processing of image stacks
from a single CLEM dataset, generating colorised composite images or image stacks for
use in the subsequent stage of the CLEM workflow.
"""

from __future__ import annotations

import argparse
from ast import literal_eval
from pathlib import Path

from cryoemservices.cli import LineWrapHelpFormatter
from cryoemservices.wrappers.clem_align_and_merge import align_and_merge_stacks


def int_or_none(value: str):
    """
    Parses the command line input, converting "null" into None and returning an
    integer otherwise.
    """
    if value.lower() == "null":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid value provided: {value}. "
            "Input must be either an integer or 'null'"
        )


def path_or_none(value: str):
    """
    Parses the command line input, converting "null" into None, existing file paths
    into Path objects, and returning None otherwise.
    """
    if value == "null":
        return None
    elif Path(value).exists():
        return Path(value)
    else:
        print(
            "The provided file path doesn't exist. Will attempt to find file using "
            "the default settings"
        )
        return None


def str_or_none(value: str):
    """
    Converts the "null" keyword into None, and returns the string as-is otherwise.
    """
    if value == "null":
        return None
    return value


def run():
    """
    Create argument parser and add arguments
    """
    parser = argparse.ArgumentParser(
        description=(
            "Takes image stacks of the colour channels from a CLEM dataset, processes "
            "them, and creates a colourised composite image or image stack for that "
            "dataset."
        ),
        formatter_class=LineWrapHelpFormatter,
    )
    # List of paths of files to merge (mandatory)
    parser.add_argument(
        dest="images",
        type=str,
        nargs="+",  # Gather as list by default and raise warning if not provided
        help=(
            "Full file paths to the image stacks to be processed in the format \n"
            "'path/to/file_1' 'path/to/file_2' ... '/path/to/file_n'"
        ),
    )
    # Metadata file (optional)
    parser.add_argument(
        "--metadata",
        default=None,
        type=path_or_none,
        help=(
            "Full file path to the metadata file associated with this dataset. "
            "If no metadata file is provided, it will attempt to find a matching "
            "metadata file at the expected default location "
            "('./Metadata/metadata_file.xml)."
        ),
    )
    # Crop image stack to the centremost N frames
    parser.add_argument(
        "--crop-to-n-frames",
        default=None,
        type=int_or_none,
        help="Crops the image stack to the centremost N frames.",
    )
    # Align image stacks before flattening
    parser.add_argument(
        "--align-self",
        default=None,
        type=str_or_none,
        help=(
            "Choose whether to align the image stacks individually before flattening. \n"
            "NOT IMPLEMENTED YET."
        ),
    )
    # Determine how the image is flattened
    parser.add_argument(
        "--flatten",
        default="mean",
        type=str_or_none,
        help=(
            "Choose whether to flatten the image stacks. \n"
            "DEFAULT: 'mean' \n"
            "VALUES: ['null', 'min', and 'max']"
        ),
    )
    # Determine what image registration protocol to implement
    parser.add_argument(
        "--align-across",
        default=None,
        type=str_or_none,
        help=(
            "Choose whether to align the image stacks to one another before merging. \n"
            "NOT IMPLEMENTED YET."
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

    """
    Run function
    """
    composite_image = align_and_merge_stacks(
        images=image_files,
        metadata=args.metadata,
        crop_to_n_frames=args.crop_to_n_frames,
        align_self=args.align_self,
        flatten=args.flatten,
        align_across=args.align_across,
        print_messages=True,  # Print messages when used as a CLI
        debug=True,  # Print debug messages
    )

    if composite_image is not None:
        print("Done")
