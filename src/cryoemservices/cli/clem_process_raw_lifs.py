from __future__ import annotations

import argparse
from pathlib import Path

from cryoemservices.cli import LineWrapHelpFormatter, set_up_logging


def run():
    # Create argument parser and needed arguments
    parser = argparse.ArgumentParser(
        description="Convert LIF files into TIFF image stacks",
        formatter_class=LineWrapHelpFormatter,
    )

    parser.add_argument(
        dest="lif_file",
        type=str,
        help="Path to LIF file for conversion",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Path to the directory that the processed data structure will be saved in. "
            "If not provided, the function will save the output file structure in the "
            "same directory as the LIF file."
        ),
    )
    parser.add_argument(
        "-n",
        "--num-procs",
        type=int,
        default=1,
        help=("Number of processes to run. \nDEFAULT:   1"),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print additional messages to check functions work as intended",
    )
    args = parser.parse_args()

    # Convert argument to correct type
    output_dir = Path(args.output_dir) if args.output_dir is not None else None

    # Set up the logger before importing the module functions
    set_up_logging(debug=args.debug)

    from cryoemservices.services.clem_process_raw_lifs import process_lif_file

    # Run function
    results = process_lif_file(
        file=Path(args.lif_file),
        output_dir=output_dir,
        number_of_processes=args.num_procs,
    )

    # Print results in output log
    if results:
        print()
        print("LIF processing workflow successfully completed")
    else:
        print()
        print("LIF processing workflow did not produce any image stacks")
