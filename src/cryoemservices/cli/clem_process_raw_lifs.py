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
        "--root-folder",
        type=str,
        default="images",
        help=(
            "Name of the top folder that LIF files are stored in. "
            "Used to determine destination of the created TIFF image stacks. \n"
            "DEFAULT:   'images'"
        ),
    )
    parser.add_argument(
        "-n",
        "--num-procs",
        type=int,
        default=1,
        help=("Number of processes to run. \n" "DEFAULT:   1"),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print additional messages to check functions work as intended",
    )
    args = parser.parse_args()

    # Set up the logger before importing the module functions
    set_up_logging(debug=args.debug)

    from cryoemservices.wrappers.clem_process_raw_lifs import process_lif_file

    # Run function
    results = process_lif_file(
        file=Path(args.lif_file),
        root_folder=args.root_folder,
        number_of_processes=args.num_procs,
    )

    # Print results in output log
    if results:
        if args.debug:
            for result in results:
                print(result)
        print()
        print("LIF processing workflow successfully completed")
    else:
        print()
        print("LIF processing workflow did not produce any image stacks")
