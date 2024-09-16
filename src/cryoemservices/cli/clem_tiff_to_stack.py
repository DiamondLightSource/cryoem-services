from __future__ import annotations

import argparse
from pathlib import Path

from cryoemservices.clem.tiff import convert_tiff_to_stack


def run():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Convert individual TIFF files into image stacks"
    )
    # Path to single TIFF file from series (Mandatory)
    parser.add_argument(
        dest="tiff_path",
        type=str,
        help="Path to any one of the TIFF files from the series to be processed",
    )
    # Root directory (Optional)
    parser.add_argument(
        "--root-dir",
        default="images",
        type=str,
        help="Top subdirectory that raw TIFF files are stored in. Used to determine destination of the created image stacks",
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
    tiff_file = Path(args.tiff_path)
    # Generate list from the single file provided
    tiff_list: list[Path] = [
        f.resolve()
        for f in tiff_file.parent.glob("./*")
        if f.suffix in {".tif", ".tiff"}
        # Handle cases where series start with the same position number,
        # but deviate afterwards
        and f.stem.startswith(tiff_file.stem.split("--")[0] + "--")
    ]

    # Resolve for metadata argument
    if not args.metadata:
        metadata = None
    else:
        metadata = Path(args.metadata)

    convert_tiff_to_stack(
        tiff_list=tiff_list,
        root_folder=args.root_dir,
        metadata_file=metadata,
    )
