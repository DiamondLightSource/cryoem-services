from __future__ import annotations

import argparse
import json
from pathlib import Path

from workflows.recipe import RecipeWrapper
from workflows.transport.pika_transport import PikaTransport

from cryoemservices.util.config import config_from_file


def run():
    parser = argparse.ArgumentParser(
        description="Resubmit a failed wrapper script using the .recipewrap file"
    )
    parser.add_argument(
        "-w",
        "--wrapper",
        help="Location of the .recipewrap wrapper file to resubmit",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config_file",
        required=True,
        help="Config file specifying the location of other credentials to read",
    )
    args = parser.parse_args()

    service_config = config_from_file(args.config_file)

    if not Path(args.wrapper).is_file():
        raise FileNotFoundError(f"{args.wrapper} cannot be found")

    # Connect to the message transport
    transport = PikaTransport()
    transport.load_configuration_file(service_config.rabbitmq_credentials)
    transport.connect()

    # Load and submit the wrapper part of the recipe
    with open(args.wrapper, "r") as wrap:
        recipe = json.load(wrap)
    rw = RecipeWrapper(message=recipe, transport=transport)
    rw._send_to_destination(rw.recipe_pointer, None, rw.payload, {})
    transport.disconnect()
