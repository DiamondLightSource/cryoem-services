import json
from pathlib import Path

from pydantic import BaseModel, ValidationError
from workflows.recipe import RecipeWrapper, wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW


class StitchMParameters(BaseModel):
    """
    These are the parameters needed by StitchM in order to convert the MRC
    image stack into a stitched atlas image.
    """

    visit_name: str
    output_dir: Path
    mosaic_mrc_file: Path
    mosaic_text_file: Path
    markers_file: Path | None = None
    normalise: bool = True
    filter: bool = True


class SIMAtlasService(CommonService):
    """
    A service that will run StitchM with the desired parameters on incoming
    SIM atlas files.
    """

    _log_name = "cryoemservices.services.sim_atlas"

    def initializing(self):
        """
        Subscribe to a queue. Received messages must be acknowledged.
        """
        self.log.info("SIM atlas stitching service starting")
        # Subscribe service to RMQ queue
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "sim.atlas",
            self.call_stitchm,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def call_stitchm(
        self,
        rw: RecipeWrapper | None,
        header: dict,
        message: dict | None,
    ):
        """
        Pass incoming message to the relevant plugin function
        """
        if not rw:
            self.log.info("Received a simple message")
            if not isinstance(message, dict):
                self.log.error("Rejected invalid simple message")
                self._reject_message(header, requeue=False)
                return

        # Create a wrapper-like object that can be passed to functions
        # as if a recipe wrapper was present
        rw = MockRW(self._transport)
        rw.recipe_step = {"parameters": message}

        try:
            if isinstance(message, dict):
                params = StitchMParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                params = StitchMParameters(**rw.recipe_step.get("parameters", {}))
        except (ValidationError, TypeError) as e:
            self.log.error(
                f"StitchMParameters validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            self._reject_message(header, transport=rw.transport, requeue=False)
            return

        self.log.info(
            "Received the following parameters:\n"
            f"{json.dumps(params.model_dump(), indent=2, default=str)}"
        )

        # File processing logic goes here

        # Ack message after completion
        self.log.info("StitchM job completed")
        rw.transport.ack(header)
        return
