from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError
from workflows.recipe import RecipeWrapper, wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW


class AlignImagesParameters(BaseModel):
    id_ref: int  # ISPyB Atlas atlasId
    image_ref: Path
    pixel_size_ref: float
    id_mov: int  # ISPyB Atlas atlasId
    image_mov: Path
    pixel_size_mov: float


class AlignImagesService(CommonService):
    """
    A CryoEM service to align to images to one another
    """

    _logger_name = __name__

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Correlative image alignment service starting")
        # Subscribe service to RMQ queue
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "correlative.align_images",
            self.call_align_images,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def call_align_images(
        self,
        rw: RecipeWrapper | None,
        header: dict[str, Any],
        message: dict[str, Any] | None,
    ):
        """Pass incoming message to the relevant plugin function."""
        # Encase message in ReceipeWrapper if none was provided
        if not rw:
            self.log.info("Received a simple message")
            if not isinstance(message, dict):
                self.log.error("Rejected invalid simple message")
                self._reject_message(header, requeue=False)
                return
            # Create a wrapper-like object to be passed to functions
            rw = MockRW(self._transport)
            rw.recipe_step = {"paramters": message}

        try:
            if isinstance(message, dict):
                params = AlignImagesParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                params = AlignImagesParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.error(
                f"AlignImagesParameters validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            self._reject_message(header, transport=rw.transport, requeue=False)
            return

        # Acknowledge receipt of parameters
        self.log.info(
            "Running image alignment with the following parameters:\n"
            f"{params.model_dump(mode='json')}"
        )

        ###############################################################################
        # Image alignment logic goes here
        ###############################################################################

        # Ack message after completion
        rw.transport.ack(header)
        return
