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


def parse_list_of_paths(values: list[str]):
    if any((not isinstance(file, str) for file in values)):
        raise TypeError("One or more of the files provided are of an invalid type")

    # Check if a stringified list has been provided (for submission in recipes)
    if len(values) == 1 and values[0].startswith("[") and values[0].endswith("]"):
        try:
            file_list: list[Path] = [
                Path(file) for file in literal_eval(values[0]) if Path(file).exists()
            ]
        except Exception:
            raise Exception("Unable to evaluate stringified list")
    # Evaluate normally as list of strings otherwise
    else:
        file_list = [Path(file) for file in values if Path(file).exists()]

    # Check that files have been found correctly
    if len(file_list) == 0:
        raise FileNotFoundError("No valid file paths provided")

    return file_list


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
            "'path/to/file_1' 'path/to/file_2' ... '/path/to/file_n'."
        ),
    )
    # Metadata file (optional)
    parser.add_argument(
        "--metadata",
        default=None,
        type=str,
        help=(
            "Full file path to the metadata file associated with this dataset. "
            "If no metadata file is provided, it will attempt to find a matching "
            "metadata file at the expected default location. "
            "('./metadata/metadata_file.xml)."
        ),
    )
    # Crop image stack to the centremost N frames
    parser.add_argument(
        "--crop-to-n-frames",
        default=None,
        type=int,
        help=(
            "Crops the image stack to the centremost N frames. If not provided, the "
            "image stack will not be cropped."
        ),
    )
    # Align image stacks before flattening
    parser.add_argument(
        "--align-self",
        default="",
        type=str,
        help=(
            "Choose whether to align the image stacks individually before flattening. \n"
            "VALUES:    'enabled', '' \n"
            "DEFAULT:   ''"
        ),
    )
    # Determine how the image is flattened
    parser.add_argument(
        "--flatten",
        default="mean",
        type=str,
        help=(
            "Choose whether to flatten the image stacks. \n"
            "VALUES:    'mean', 'min', 'max', '' \n"
            "DEFAULT:   'mean'"
        ),
    )
    # Determine what image registration protocol to implement
    parser.add_argument(
        "--align-across",
        default="",
        type=str,
        help=(
            "Choose whether to align the image stacks to one another before merging. \n"
            "VALUES:    'enabled', '' \n"
            "DEFAULT:   ''"
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

    # Validate image stacks parameter
    image_files: list[Path] = parse_list_of_paths(args.images)
    print("Found all provided image files")
    if args.debug:
        [print(file) for file in image_files]

    # Convert metadata argument into a Path
    metadata = (
        (Path(args.metadata) if args.metadata else None)
        if isinstance(args.metadata, str)
        else args.metadata
    )

    """
    Run function
    """
    composite_image = align_and_merge_stacks(
        images=image_files,
        metadata=metadata,
        crop_to_n_frames=args.crop_to_n_frames,
        align_self=args.align_self,
        flatten=args.flatten,
        align_across=args.align_across,
        print_messages=True,  # Print messages when used as a CLI
        debug=args.debug,  # Print debug messages
    )

    if composite_image is not None:
        print("Done")
