from __future__ import annotations

import argparse
import json
from pathlib import Path

import workflows.transport.pika_transport as pt
from workflows.recipe import RecipeWrapper


def run():
    parser = argparse.ArgumentParser(
        description="Resubmit a failed zocalo wrapper script using the .recipewrap file"
    )
    parser.add_argument(
        "-w",
        "--wrapper",
        help="Location of the .recipewrap wrapper file to resubmit",
        dest="wrapper",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Transport configuration file for connecting to the message broker",
        dest="config",
        required=True,
    )
    args = parser.parse_args()

    if not Path(args.wrapper).is_file():
        print(f"{args.wrapper} cannot be found")
        return
    if not Path(args.config).is_file():
        print(f"{args.config} cannot be found")
        return

    # Connect to the message transport
    transport = pt.PikaTransport()
    transport.load_configuration_file(args.config)
    transport.connect()

    # Load and submit the wrapper part of the recipe
    with open(args.wrapper, "r") as wrap:
        recipe = json.load(wrap)
    rw = RecipeWrapper(message=recipe, transport=transport)
    rw._send_to_destination(rw.recipe_pointer, None, rw.payload, {})


if __name__ == "__main__":
    run()
