from __future__ import annotations

import time

import ispyb.sqlalchemy
import sqlalchemy.orm
import workflows.recipe
from workflows.services.common_service import CommonService

from cryoemservices.util import ispyb_commands
from cryoemservices.util.config import config_from_file
from cryoemservices.util.models import MockRW


class EMISPyB(CommonService):
    """A service that receives information to be written to ISPyB."""

    # Human readable service name
    _service_name = "EMISPyB"

    # Logger name
    _logger_name = "cryoemservices.services.ispyb_connector"

    # ispyb connection details
    ispyb = None
    _ispyb_sessionmaker = None

    def initializing(self):
        """Subscribe the ISPyB connector queue. Received messages must be
        acknowledged. Prepare ISPyB database connection."""
        service_config = config_from_file(self._environment["config"])
        self._ispyb_sessionmaker = sqlalchemy.orm.sessionmaker(
            bind=sqlalchemy.create_engine(
                ispyb.sqlalchemy.url(credentials=service_config.ispyb_credentials),
                connect_args={"use_pure": True},
            )
        )
        self.log.info("ISPyB service ready")
        workflows.recipe.wrap_subscribe(
            self._transport,
            "ispyb_connector",
            self.insert_into_ispyb,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def insert_into_ispyb(self, rw, header, message):
        """Do something with ISPyB."""
        if not rw:
            # Incoming message is not a recipe message. Simple messages can be valid
            self.log.info("Received a simple message")
            if (
                not isinstance(message, dict)
                or not message.get("parameters")
                or not message.get("content")
            ):
                self.log.error("Rejected invalid simple message")
                self._transport.nack(header)
                return

            # Create a wrapper-like object that can be passed to functions
            # as if a recipe wrapper was present.
            rw = MockRW(self._transport)
            rw.recipe_step = {"parameters": message["parameters"]}
            if isinstance(message["content"], dict) and isinstance(
                message["parameters"], dict
            ):
                message["content"].update(message["parameters"])
            message = message["content"]

        if not message:
            # Message must be a dictionary, but at this point it could be None or ""
            message = {}
        elif type(message) is str:
            message = {"status_message": message}

        def parameters(parameter):
            if message.get(parameter):
                base_value = message[parameter]
            else:
                base_value = rw.recipe_step["parameters"].get(parameter)
            if (
                not base_value
                or not isinstance(base_value, str)
                or "$" not in base_value
            ):
                return base_value
            for key in sorted(rw.environment, key=len, reverse=True):
                if "${" + str(key) + "}" in base_value:
                    base_value = base_value.replace(
                        "${" + str(key) + "}", str(rw.environment[key])
                    )
                # Replace longest keys first, as the following replacement is
                # not well-defined when one key is a prefix of another:
                if f"${key}" in base_value:
                    base_value = base_value.replace(f"${key}", str(rw.environment[key]))
            return base_value

        command = parameters("ispyb_command")
        if not command:
            self.log.error("Received message is not a valid ISPyB command")
            rw.transport.nack(header)
            return
        command_function = getattr(ispyb_commands, command, None)
        if not command_function:
            self.log.error("Received unknown ISPyB command (%s)", command)
            rw.transport.nack(header)
            return

        # Set an expiry time for this message, for delays on database synchronisation
        if not message.get("expiry_time") or header.get("dlq-reinjected"):
            message["expiry_time"] = time.time() + 600

        self.log.info("Running ISPyB call %s", command)
        try:
            with self._ispyb_sessionmaker() as session:
                result = command_function(
                    message=message,
                    parameters=parameters,
                    session=session,
                )
        except Exception as e:
            self.log.error(
                f"Uncaught exception {e!r} in ISPyB function {command!r}, "
                "quarantining message and shutting down instance.",
                exc_info=True,
            )
            rw.transport.nack(header)
            self._request_termination()
            return

        # Store results if they are requested in the parameters or in the command
        if result:
            store_result_global = parameters("store_result")
            store_result_local = result.get("store_result")
            if store_result_local:
                rw.environment[store_result_local] = result["return_value"]
                self.log.info(
                    f"Storing {result['return_value']} in {store_result_local}"
                )
            if store_result_global:
                rw.environment[store_result_global] = result["return_value"]
                self.log.info(
                    f"Storing {result['return_value']} in {store_result_global}"
                )

        if result and result.get("success"):
            rw.set_default_channel("output")
            rw.send({"result": result.get("return_value")})
            rw.transport.ack(header)
        elif result and result.get("checkpoint"):
            rw.checkpoint(result.get("checkpoint_dict"))
            rw.transport.ack(header)
        elif message["expiry_time"] > time.time():
            self.log.warning(
                f"Failed call {command}. Checkpointing for delayed resubmission"
            )
            rw.checkpoint(message, delay=20)
            rw.transport.ack(header)
        else:
            self.log.error(f"ISPyB request for {command} failed")
            rw.transport.nack(header)
