from __future__ import annotations

import argparse
from pathlib import Path

from cryoemservices.wrappers.clem_process_raw_lifs import convert_lif_to_stack


def run():
    # Create argument parser and needed arguments
    parser = argparse.ArgumentParser(
        description="Convert LIF files into TIFF image stacks"
    )

    parser.add_argument(
        dest="lif_path",
        type=str,
        help="Path to LIF file for conversion",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="images",
        help="Top subdirectory that LIF files are stored in. Used to determine destination of the created TIFF image stacks",
    )
    parser.add_argument(
        "-n",
        "--num-procs",
        type=int,
        default=1,
        help="Number of processes to run",
    )
    args = parser.parse_args()

    # Load args
    lif_path = Path(args.lif_path)
    root_folder: str = args.root_dir
    num_procs: int = args.num_procs

    # Run function
    results = convert_lif_to_stack(
        file=lif_path,
        root_folder=root_folder,
        number_of_processes=num_procs,
    )

    # Print results in output log
    if results is not None:
        for result in results:
            print(result)
