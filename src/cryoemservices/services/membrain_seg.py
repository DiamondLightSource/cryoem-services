from __future__ import annotations

from pathlib import Path
from typing import Optional

import workflows.recipe
from pydantic import BaseModel, Field, ValidationError
from workflows.services.common_service import CommonService

from cryoemservices.util.slurm_submission import slurm_submission


class MembrainSegParameters(BaseModel):
    tomogram: str = Field(..., min_length=1)
    output_dir: Optional[str] = None
    model_checkpoint: str = (
        "/dls_sw/apps/EM/membrain-seg/models/MemBrain_seg_v10_alpha.ckpt"
    )
    pixel_size: Optional[float] = None
    rescale_patches: bool = True
    augmentation: bool = False
    store_probabilities: bool = False
    store_connected_components: bool = False
    window_size: int = 160
    connected_component_threshold: Optional[int] = None
    segmentation_threshold: Optional[float] = None
    cleanup_output: bool = True


class MembrainSeg(CommonService):
    """
    A service for segmenting cryoEM tomograms using membrain-seg
    """

    # Human readable service name
    _service_name = "MembrainSeg"

    # Logger name
    _logger_name = "cryoemservices.services.membrain_seg"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("membrain-seg service starting")
        workflows.recipe.wrap_subscribe(
            self._transport,
            "segmentation",
            self.membrain_seg,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def membrain_seg(self, rw, header: dict, message: dict):
        class MockRW:
            def dummy(self, *args, **kwargs):
                pass

        if not rw:
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
            rw = MockRW()
            rw.transport = self._transport
            rw.recipe_step = {"parameters": message["parameters"], "output": None}
            rw.environment = {"has_recipe_wrapper": False}
            rw.set_default_channel = rw.dummy
            rw.send = rw.dummy
            message = message["content"]

        try:
            if isinstance(message, dict):
                membrain_seg_params = MembrainSegParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                membrain_seg_params = MembrainSegParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"membrain-seg parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        # Assemble the membrain-seg command
        if not membrain_seg_params.output_dir:
            segmented_output_dir = Path(membrain_seg_params.tomogram).parent
        else:
            segmented_output_dir = Path(membrain_seg_params.output_dir)
        segmented_output_dir.mkdir(exist_ok=True, parents=True)
        command = ["membrain", "segment", "--out-folder", str(segmented_output_dir)]

        membrain_seg_flags = {
            "tomogram": "--tomogram-path",
            "model_checkpoint": "--ckpt-path",
            "pixel_size": "--in-pixel-size",
            "connected_component_threshold": "--connected-component-thres",
            "segmentation_threshold": "--segmentation-threshold",
            "window_size": "--sliding-window-size",
        }
        for k, v in membrain_seg_params.dict().items():
            if v and (k in membrain_seg_flags):
                command.extend((membrain_seg_flags[k], str(v)))

        if membrain_seg_params.rescale_patches:
            command.append("--rescale-patches")
        else:
            command.append("--no-rescale-patches")

        if membrain_seg_params.augmentation:
            command.append("--test-time-augmentation")
        else:
            command.append("--no-test-time-augmentation")

        if membrain_seg_params.store_probabilities:
            command.append("--store-probabilities")
        else:
            command.append("--no-store-probabilities")

        if membrain_seg_params.store_connected_components:
            command.append("--store-connected-components")
        else:
            command.append("--no-store-connected-components")

        # Determine the output paths
        segmented_file = f"{Path(membrain_seg_params.tomogram).stem}_segmented.mrc"
        segmented_path = segmented_output_dir / segmented_file

        membrain_file = (
            f"{Path(membrain_seg_params.tomogram).stem}"
            f"_{Path(membrain_seg_params.model_checkpoint).name}_segmented.mrc"
        )
        membrain_path = segmented_output_dir / membrain_file

        self.log.info(f"Input: {membrain_seg_params.tomogram} Output: {segmented_path}")
        self.log.info(f"Running {command}")

        # Submit the command to slurm
        result = slurm_submission(
            log=self.log,
            job_name="membrain-seg",
            command=command,
            project_dir=segmented_output_dir,
            output_file=segmented_path,
            cpus=1,
            use_gpu=True,
            use_singularity=False,
            memory_request=25000,
            script_extras="module load EM/membrain-seg",
        )

        # Run the command
        # result = subprocess.run(command, capture_output=True)

        # Stop here if the job failed
        if result.returncode:
            self.log.error("membrain-seg failed to run")
            rw.transport.nack(header)
            return

        # Rename the output file
        if membrain_path.is_file():
            membrain_path.rename(segmented_path)

        # Clean up the slurm files
        if membrain_seg_params.cleanup_output:
            Path(f"{segmented_path}.out").unlink()
            Path(f"{segmented_path}.err").unlink()
            Path(f"{segmented_path}.json").unlink()

        # Forward results to images service
        self.log.info(f"Sending to images service {segmented_path}")
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="images",
                message={
                    "image_command": "mrc_central_slice",
                    "file": str(segmented_path),
                    "skip_rescaling": True,
                },
            )
            rw.transport.send(
                destination="movie",
                message={
                    "image_command": "mrc_to_apng",
                    "file": str(segmented_path),
                    "skip_rescaling": True,
                },
            )
        else:
            rw.send_to(
                "images",
                {
                    "image_command": "mrc_central_slice",
                    "file": str(segmented_path),
                    "skip_rescaling": True,
                },
            )
            rw.send_to(
                "movie",
                {
                    "image_command": "mrc_to_apng",
                    "file": str(segmented_path),
                    "skip_rescaling": True,
                },
            )

        # Insert the segmented tomogram into ISPyB
        ispyb_parameters = {
            "ispyb_command": "insert_processed_tomogram",
            "file_path": str(segmented_path),
            "processing_type": "Segmented",
        }
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="ispyb_connector",
                message={
                    "parameters": ispyb_parameters,
                    "content": {"dummy": "dummy"},
                },
            )
        else:
            rw.send_to("ispyb_connector", ispyb_parameters)

        self.log.info(f"Done segmentation for {membrain_seg_params.tomogram}")
        rw.transport.ack(header)
        return
