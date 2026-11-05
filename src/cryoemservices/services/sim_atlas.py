import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator
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

    @field_validator("markers_file", mode="before")
    @classmethod
    def parse_markers_file(cls, value):
        # Convert 'null' and 'None' strings into 'None' object
        if isinstance(value, str) and value in ("null", "None"):
            return None
        # Let Pydantic validate it as-is otherwise
        return value


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

        # Create a temp directory to save the intermediate results to
        tmp_dir = params.output_dir / str(uuid.uuid4())
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Create a new copy of the 'mosaic.txt' file with the correct file path
        mosaic_text_file_new = tmp_dir / params.mosaic_text_file.name
        mosaic_text_file_new.parent.mkdir(parents=True, exist_ok=True)
        with open(params.mosaic_text_file, "r") as old:
            with open(mosaic_text_file_new, "w") as new:
                for l, line in enumerate(old.readlines()):
                    # The first line contains the file path on the source side
                    # This should be replaced with the file path on the server
                    if l == 0:
                        new.write(str(params.mosaic_mrc_file) + "\n")
                    else:
                        new.write(line)

        # Run StitchM on the new mosaic text file
        output_file: Path | None = None
        cmd = [
            "stitchm",
            "--mosaic",
            str(mosaic_text_file_new),
        ]
        if params.markers_file is not None:
            cmd.extend(["--markers", str(params.markers_file)])
        if not params.filter:
            cmd.extend(["--unfiltered"])
        if not params.normalise:
            cmd.extend(["--no-normalisation"])
        process_result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge the streams
            text=True,
        )
        # Attempt to extract output file from subprocess logs
        for line in process_result.stdout.splitlines():
            line = line.rstrip()
            if line.startswith("INFO     Saving") and line.endswith("ome.tiff"):
                output_file = Path(line.split("INFO     Saving")[1].strip())
        # Check process return code
        if return_code := process_result.returncode:
            self.log.error(
                f"StitchM subprocess failed with error code {return_code}"
                # Log any stdout as part of the error
                + (f":\n{process_result.stdout}" if process_result.stdout else "")
            )
            shutil.rmtree(tmp_dir)
            self._reject_message(header, transport=rw.transport, requeue=False)
            return
        # Mark as failure and reject message if no output file was found
        if not output_file or not output_file.is_file():
            self.log.error(
                f"PySIMRecon failed to generate output file for {params.file}"
                # Log any stdout as part of the error
                + (f":\n{process_result.stdout}" if process_result.stdout else "")
            )
            shutil.rmtree(tmp_dir)
            self._reject_message(header, transport=rw.transport, requeue=False)
            return
        # Log stdout as one block if successful
        self.log.info(process_result.stdout)

        # Move the file out of the UUID folder and delete the old directory
        output_file_new = params.output_dir / output_file.name
        shutil.move(output_file, output_file_new)
        shutil.rmtree(tmp_dir)
        self.log.info(f"Stitched atlas image saved to {output_file_new}")

        # Construct message to send back to Murfey
        result: dict[str, Any] = {
            "output_file": str(output_file_new),
            # More fields can be added as workflow develops
        }
        murfey_params = {
            "register": "sim.register_atlas",
            "result": result,
        }
        self.log.info(
            "Will submit the following message back to Murfey:\n"
            f"{json.dumps(murfey_params, indent=2, default=str)}"
        )
        # rw.send_to("murfey_feedback", murfey_params)  # Will be enabled in the future

        # Ack message after completion
        self.log.info("StitchM job completed")
        rw.transport.ack(header)
        return
