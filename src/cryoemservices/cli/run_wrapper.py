from __future__ import annotations

import argparse
import json
import logging

import graypy
from backports.entry_points_selectable import entry_points
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.pika_transport import PikaTransport

from cryoemservices.util.config import config_from_file


def run():
    known_wrappers = {
        e.name: e.load for e in entry_points(group="cryoemservices.wrappers")
    }

    # Parse command line arguments
    parser = argparse.ArgumentParser(usage="cryoemservices.wrap [options]")
    parser.add_argument(
        "-w",
        "--wrap",
        action="store",
        dest="wrapper",
        required=True,
        choices=list(known_wrappers),
        help="Object to be wrapped. Known wrappers: " + ", ".join(known_wrappers),
    )
    parser.add_argument(
        "-r",
        "--recipewrapper",
        action="store",
        dest="recipewrapper",
        required=True,
        help="A serialized recipe wrapper file for downstream communication",
    )
    parser.add_argument(
        "-c",
        "--config_file",
        action="store",
        required=True,
        help="Config file",
    )
    args = parser.parse_args()

    service_config = config_from_file(args.config_file)

    # Initialize logging
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    if service_config.graylog_host:
        graylog_handler = graypy.GELFUDPHandler(
            service_config.graylog_host, service_config.graylog_port, level_names=True
        )
        logging.getLogger().addHandler(graylog_handler)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("pika").setLevel(logging.WARN)
    log = logging.getLogger("cryoemservices.wrap")

    log.info(
        f"Starting wrapper for {args.wrapper} "
        f"with recipewrapper file {args.recipewrapper}",
    )

    # Connect to transport and start sending notifications
    transport = PikaTransport()
    transport.load_configuration_file(service_config.rabbitmq_credentials)
    transport.connect()

    # Instantiate chosen wrapper
    instance = known_wrappers[args.wrapper]()()

    # If specified, read in a serialized recipewrapper
    with open(args.recipewrapper) as fh:
        recwrap = RecipeWrapper(message=json.load(fh), transport=transport)
    instance.set_recipe_wrapper(recwrap)

    instance.prepare({"status_message": "Starting processing"})
    log.info("Setup complete, starting processing")
    try:
        if instance.run():
            log.info("successfully finished processing")
            instance.success(
                {"status_message": "Finished processing", "status": "success"}
            )
        else:
            log.info("processing failed")
            instance.failure(
                {"status_message": "Processing failed", "status": "failure"}
            )
    except KeyboardInterrupt:
        log.info("Shutdown via Ctrl+C")
    except Exception as e:
        log.error(str(e), exc_info=True)
        instance.failure(
            {"status_message": "Exception in processing", "status": "failure"}
        )
    log.info("Wrapper terminating")
    transport.disconnect()