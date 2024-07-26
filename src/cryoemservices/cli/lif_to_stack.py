from __future__ import annotations

import argparse
from pathlib import Path

from cryoemservices.clem.lif import convert_lif_to_stack


def run():
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

    convert_lif_to_stack(
        file=Path(args.lif_path),
        root_folder=args.root_dir,
        number_of_processes=args.num_procs,
    )
