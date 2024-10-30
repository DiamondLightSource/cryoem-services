from __future__ import annotations

import argparse
from pathlib import Path

from cryoemservices.wrappers.clem_process_raw_tiffs import convert_tiff_to_stack


def run():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Convert individual TIFF files into image stacks"
    )
    # Path to single TIFF file from series (Mandatory)
    parser.add_argument(
        dest="tiff_file",
        type=str,
        help="Path to any one of the TIFF files from the series to be processed",
    )
    # Root directory (Optional)
    parser.add_argument(
        "--root-folder",
        default="images",
        type=str,
        help="Name of the top folder that raw TIFF files are stored in. Used to determine destination of the created image stacks",
    )
    # Path to metadata file (Optional)
    parser.add_argument(
        "--metadata",
        default=None,
        type=str,
        help="Path to the XLIF file associated with this dataset. If not provided, the script will use relative file paths to find what it thinks is the appropriate file",
    )
    # Parse the arguments
    args = parser.parse_args()

    # Convert to correct object types
    tiff_file = Path(args.tiff_file)
    # Generate list from the single file provided
    tiff_list: list[Path] = [
        f.resolve()
        for f in tiff_file.parent.glob("./*")
        if f.suffix in {".tif", ".tiff"}
        # Handle cases where series start with the same position number,
        # but deviate afterwards
        and f.stem.startswith(tiff_file.stem.split("--")[0] + "--")
    ]

    # Parse root folder argument
    root_folder: str = args.root_folder

    # Parse metadata argument
    if not args.metadata:
        metadata = None
    else:
        metadata = Path(args.metadata)

    result = convert_tiff_to_stack(
        tiff_list=tiff_list,
        root_folder=root_folder,
        metadata_file=metadata,
    )

    # Print result to output log
    if result is not None:
        print(result)
