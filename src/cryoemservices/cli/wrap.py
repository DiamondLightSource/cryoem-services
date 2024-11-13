from __future__ import annotations

import argparse
import json
import logging
from importlib.metadata import entry_points

from workflows.recipe.wrapper import RecipeWrapper
from workflows.services import common_service
from workflows.transport.pika_transport import PikaTransport
from zocalo.wrapper import StatusNotifications

from cryoemservices.util.config import config_from_file


def run():
    known_wrappers = {
        e.name: e.load for e in entry_points(group="cryoemservices.wrappers")
    }

    # Parse command line arguments
    parser = argparse.ArgumentParser(usage="cryoemservices.wrap [options]")
    parser.add_argument(
        "--wrap",
        action="store",
        dest="wrapper",
        required=True,
        choices=list(known_wrappers),
        help="Object to be wrapped. Known wrappers: " + ", ".join(known_wrappers),
    )
    parser.add_argument(
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

    # Initialize logging
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    logging.getLogger().setLevel(logging.INFO)
    log = logging.getLogger("zocalo.wrap")

    log.info(
        f"Starting wrapper for {args.wrapper} "
        f"with recipewrapper file {args.recipewrapper}",
    )

    service_config = config_from_file(args.config_file)

    # Connect to transport and start sending notifications
    transport = PikaTransport()
    transport.load_configuration_file(service_config.rabbitmq_credentials)
    transport.connect()
    st = StatusNotifications(transport.broadcast_status, args.wrapper)

    # Instantiate chosen wrapper
    instance = known_wrappers[args.wrapper]()()
    instance.status_thread = st

    # If specified, read in a serialized recipewrapper
    with open(args.recipewrapper) as fh:
        recwrap = RecipeWrapper(message=json.load(fh), transport=transport)
    instance.set_recipe_wrapper(recwrap)

    if recwrap.recipe_step.get("wrapper", {}).get("task_information"):
        # Add any extra task_information field to the status display
        st.taskname += (
            " (" + str(recwrap.recipe_step["wrapper"]["task_information"]) + ")"
        )

    instance.prepare("Starting processing")
    st.set_status(common_service.Status.PROCESSING)
    log.info("Setup complete, starting processing")

    try:
        if instance.run():
            log.info("successfully finished processing")
            instance.success("Finished processing")
        else:
            log.info("processing failed")
            instance.failure("Processing failed")
        st.set_status(common_service.Status.END)
    except KeyboardInterrupt:
        log.info("Shutdown via Ctrl+C")
        st.set_status(common_service.Status.END)
    except Exception as e:
        log.error(str(e), exc_info=True)
        instance.failure(e)
        st.set_status(common_service.Status.ERROR)

    instance.done("Finished processing")
    st.shutdown()
    st.join()
    log.debug("Terminating")
    transport.disconnect()
