from __future__ import annotations

import argparse
from pathlib import Path

from cryoemservices.cli import LineWrapHelpFormatter, set_up_logging


def run():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Convert individual TIFF files into image stacks",
        formatter_class=LineWrapHelpFormatter,
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
    # Add debug option
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print additional messages to check functions work as intended",
    )
    # Parse the arguments
    args = parser.parse_args()

    # Configure logger to log to console
    set_up_logging(debug=args.debug)

    # Import module only after logger has been set up
    from cryoemservices.wrappers.clem_process_raw_tiffs import process_tiff_files

    # Generate list from the single file provided
    tiff_file = Path(args.tiff_file)
    tiff_list: list[Path] = [
        f.resolve()
        for f in tiff_file.parent.glob("./*")
        if f.suffix in {".tif", ".tiff"}
        # Handle cases where series start with the same position number,
        # but deviate afterwards
        and f.stem.startswith(tiff_file.stem.split("--")[0] + "--")
    ]

    # Parse metadata argument
    metadata = None if not args.metadata else Path(args.metadata)

    results = process_tiff_files(
        tiff_list=tiff_list,
        root_folder=args.root_folder,
        metadata_file=metadata,
    )

    # Print result to output log
    if results:
        if args.debug:
            for result in results:
                print(result)
        print()
        print("TIFF processing workflow successfully completed")
    else:
        print()
        print("TIFF procesisng workflow did not produce any image stacks")
