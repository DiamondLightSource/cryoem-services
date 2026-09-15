from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, ValidationError, field_validator
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import RelionServiceOptions

try:
    import torch
    from topaz.denoise import Denoise3D, denoise_tomogram_stream

    run_subprocess = False
except ImportError:
    run_subprocess = True


class DenoiseParameters(BaseModel):
    volume: str = Field(..., min_length=1)
    output_dir: str | None = None  # volume directory
    suffix: str = ".denoised"
    model: str = "unet-3d"
    gaussian: int = 0
    patch_size: int = 96
    patch_padding: int = 48
    cleanup_output: bool = True
    copy_output: bool = False
    visits_for_slurm: list | None = ["bi", "cm", "nr", "nt"]
    relion_options: RelionServiceOptions

    @field_validator("model")
    @classmethod
    def saved_models(cls, v):
        if v not in ["unet-3d-10a", "unet-3d-20a", "unet-3d"]:
            raise ValueError("Model must be one of unet-3d-10a, unet-3d-20a, unet-3d")
        return v


class Denoise(CommonService):
    """
    A service for denoising cryoEM tomograms using Topaz
    """

    # Logger name
    _logger_name = "cryoemservices.services.denoise"

    # Job name
    job_type = "relion.denoisetomo"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Denoise service starting")
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "denoise",
            self.denoise,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    @staticmethod
    def check_visit(denoise_params: DenoiseParameters):
        return True

    def run_topaz(
        self,
        topaz_command: List[str],
        alignment_output_dir: Path,
        denoise_parameters: DenoiseParameters,
        denoised_full_path: Path,
    ):
        if run_subprocess:
            self.log.info("Running topaz through subprocess")
            return subprocess.run(topaz_command, capture_output=True)

        torch.set_num_threads(1)
        torch.cuda.set_device(0)

        denoiser = Denoise3D(denoise_parameters.model, True, dims=3)
        denoised_volumes = denoise_tomogram_stream(
            volumes=[denoise_parameters.volume],
            model=denoiser,
            output_path=str(alignment_output_dir),
            suffix=denoise_parameters.suffix,
            gaus=denoise_parameters.gaussian,
            patch_size=denoise_parameters.patch_size,
            padding=denoise_parameters.patch_padding,
            verbose=True,
            use_cuda=True,
        )
        if len(denoised_volumes) == 1:
            rtc = 0
        else:
            rtc = 1
        return subprocess.CompletedProcess(
            args="",
            returncode=rtc,
            stdout="".encode("utf8"),
            stderr="".encode("utf8"),
        )

    def denoise(self, rw, header: dict, message: dict):
        """Main function which interprets and processes received messages"""
        if not rw:
            self.log.info("Received a simple message")
            if not isinstance(message, dict):
                self.log.error("Rejected invalid simple message")
                self._reject_message(header, requeue=False)
                return

            # Create a wrapper-like object that can be passed to functions
            # as if a recipe wrapper was present.
            rw = MockRW(self._transport)
            rw.recipe_step = {"parameters": message}

        try:
            if isinstance(message, dict):
                denoise_params = DenoiseParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                denoise_params = DenoiseParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Denoise parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            self._reject_message(header, transport=rw.transport, requeue=False)
            return

        if not self.check_visit(denoise_params):
            # This one should infinitely nack
            self.log.warning(f"Visit rejected for {denoise_params.volume}")
            rw.transport.nack(header, requeue=True)
            return

        command = [
            "topaz",
            "denoise3d",
            denoise_params.volume,
        ]

        denoise_flags = {
            "output_dir": "-o",
            "suffix": "--suffix",
            "model": "-m",
            "gaussian": "-g",
            "patch_size": "-s",
            "patch_padding": "-p",
        }
        for k, v in denoise_params.model_dump().items():
            if (v not in [None, ""]) and (k in denoise_flags):
                command.extend((denoise_flags[k], str(v)))

        if denoise_params.output_dir:
            Path(denoise_params.output_dir).mkdir(parents=True, exist_ok=True)
            alignment_output_dir = Path(denoise_params.output_dir)
        else:
            alignment_output_dir = Path(denoise_params.volume).parent

        suffix = str(Path(denoise_params.volume).suffix)
        denoised_file = (
            str(Path(denoise_params.volume).stem) + denoise_params.suffix + suffix
        )
        denoised_full_path = alignment_output_dir / denoised_file
        if denoised_full_path.is_file():
            job_is_rerun = True
        else:
            job_is_rerun = False

        # Run topaz either locally or using Slurm
        self.log.info(f"Input: {denoise_params.volume} Output: {denoised_full_path}")
        result = self.run_topaz(
            topaz_command=command,
            alignment_output_dir=alignment_output_dir,
            denoise_parameters=denoise_params,
            denoised_full_path=denoised_full_path,
        )

        if not job_is_rerun:
            # Send to node creator if this is the first time this tomogram is made
            self.log.info("Sending denoising to node creator")
            node_creator_parameters = {
                "experiment_type": "tomography",
                "job_type": self.job_type,
                "input_file": denoise_params.volume,
                "output_file": str(denoised_full_path),
                "relion_options": dict(denoise_params.relion_options),
                "command": " ".join(command),
                "stdout": result.stdout.decode("utf8", "replace"),
                "stderr": result.stderr.decode("utf8", "replace"),
                "success": True,
            }
            if result.returncode:
                node_creator_parameters["success"] = False
            rw.send_to("node_creator", node_creator_parameters)

        # Stop here if the job failed
        if result.returncode:
            self.log.error("Denoising failed to run")
            self._reject_message(header, transport=rw.transport)
            return

        # Clean up the slurm files
        if denoise_params.cleanup_output:
            Path(f"{denoised_full_path}.out").unlink(missing_ok=True)
            Path(f"{denoised_full_path}.err").unlink(missing_ok=True)
            Path(f"{denoised_full_path}.json").unlink(missing_ok=True)

        # Forward results to images service
        self.log.info(f"Sending to images service {denoise_params.volume}")
        rw.send_to(
            "images",
            {
                "image_command": "mrc_central_slice",
                "file": str(denoised_full_path),
            },
        )
        rw.send_to(
            "movie",
            {
                "image_command": "mrc_to_apng",
                "file": str(denoised_full_path),
            },
        )

        # Send to segmentation and picking
        self.log.info(f"Sending {denoised_full_path} for segmentation and picking")
        if denoise_params.output_dir:
            project_dir_search = re.search(".+/job[0-9]+/", denoise_params.output_dir)
            job_num_search = re.search("/job[0-9]+", denoise_params.output_dir)
            if project_dir_search and job_num_search:
                project_dir = Path(project_dir_search[0]).parent.parent
                job_number = int(job_num_search[0][4:])
                segmentation_dir = (
                    project_dir / f"Segmentation/job{job_number + 1:03}/tomograms"
                )
                cryolo_dir = project_dir / f"AutoPick/job{job_number + 2:03}"
            else:
                self.log.warning(f"No job number in {denoise_params.output_dir}")
                segmentation_dir = Path(denoise_params.output_dir)
                cryolo_dir = Path(denoise_params.output_dir)
        else:
            segmentation_dir = Path(denoise_params.volume).parent
            cryolo_dir = Path(denoise_params.volume).parent
        segmentation_parameters = {
            "tomogram": str(denoised_full_path),
            "output_dir": str(segmentation_dir),
            "pixel_size": str(denoise_params.relion_options.pixel_size_downscaled),
            "copy_output": denoise_params.copy_output,
            "relion_options": dict(denoise_params.relion_options),
        }
        cryolo_parameters = {
            "input_path": str(denoised_full_path),
            "output_path": str(cryolo_dir / f"CBOX_3D/{denoised_full_path.stem}.cbox"),
            "experiment_type": "tomography",
            "cryolo_box_size": 40,
            "relion_options": dict(denoise_params.relion_options),
        }
        rw.send_to("segmentation", segmentation_parameters)
        rw.send_to("cryolo", cryolo_parameters)

        # Insert the denoised tomogram into ISPyB
        ispyb_parameters = {
            "ispyb_command": "insert_processed_tomogram",
            "file_path": str(denoised_full_path),
            "processing_type": "Denoised",
        }
        rw.send_to("ispyb_connector", ispyb_parameters)

        # Optionally copy output file
        if denoise_params.copy_output:
            # Take file name for Relion-type projects, or folder name for SXT-style
            tomo_name = (
                denoised_full_path.name
                if re.match(".*/job[0-9]+/.*", str(denoised_full_path))
                else f"{denoised_full_path.parent.parent}_denoised.mrc"
            )
            shutil.copy(
                denoised_full_path, denoised_full_path.parent.parent.parent / tomo_name
            )

        self.log.info(f"Done denoising for {denoise_params.volume}")
        rw.transport.ack(header)
        return
