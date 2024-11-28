from __future__ import annotations

import ispyb.sqlalchemy
import sqlalchemy.orm
import workflows.recipe
from workflows.services.common_service import CommonService

from cryoemservices.util import ispyb_commands
from cryoemservices.util.models import MockRW


class EMISPyB(CommonService):
    """A service that receives information to be written to ISPyB."""

    # Human readable service name
    _service_name = "EMISPyB"

    # Logger name
    _logger_name = "cryoemservices.services.ispyb"

    # ispyb connection details
    ispyb = None
    _ispyb_sessionmaker = None

    def initializing(self):
        """Subscribe the ISPyB connector queue. Received messages must be
        acknowledged. Prepare ISPyB database connection."""
        self._ispyb_sessionmaker = sqlalchemy.orm.sessionmaker(
            bind=sqlalchemy.create_engine(
                ispyb.sqlalchemy.url(), connect_args={"use_pure": True}
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

        def parameters(parameter):
            if isinstance(message, dict) and parameter in message:
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

        store_result = rw.recipe_step["parameters"].get("store_result")
        if result and isinstance(result.get("return_value"), dict):
            store_checkpointed_result = result["return_value"].get("store_result")
            store_checkpoint_value = result["return_value"].get("store_value")
            if store_result:
                rw.environment[store_result] = result["return_value"]
                self.log.info(
                    f"Storing {result['return_value']} in "
                    f"environment variable {store_result}.",
                )
            if store_checkpointed_result and store_checkpoint_value:
                rw.environment[store_checkpointed_result] = store_checkpoint_value
                self.log.info(
                    f"Storing checkpointed value {store_checkpoint_value} in "
                    f"environment variable {store_checkpointed_result}.",
                )

        if result and result.get("success"):
            if isinstance(rw, MockRW):
                rw.transport.send(
                    destination="output",
                    message={"result": result.get("return_value")},
                )
            else:
                rw.send_to(
                    "output",
                    {"result": result.get("return_value")},
                )
            rw.transport.ack(header)
        elif result and result.get("checkpoint"):
            rw.checkpoint(result.get("return_value"))
            rw.transport.ack(header)
        else:
            rw.transport.nack(header)
            return
