from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Optional

import workflows.recipe
from pydantic import BaseModel, Field, ValidationError, field_validator
from workflows.services.common_service import CommonService

from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import RelionServiceOptions


class DenoiseParameters(BaseModel):
    volume: str = Field(..., min_length=1)
    output_dir: Optional[str] = None  # volume directory
    suffix: str = ".denoised"
    model: Optional[str] = None  # "unet-3d"
    even_train_path: Optional[str] = None
    odd_train_path: Optional[str] = None
    n_train: Optional[int] = None  # 1000
    n_test: Optional[int] = None  # 200
    crop: Optional[int] = None  # 96
    base_kernel_width: Optional[int] = None  # 11
    optim: Optional[str] = None  # "adagrad"
    lr: Optional[float] = None  # 0.001
    criteria: Optional[str] = None  # "L2"
    momentum: Optional[float] = None  # 0.8
    batch_size: Optional[int] = None  # 10
    num_epochs: Optional[int] = None  # 500
    weight_decay: Optional[int] = None  # 0
    save_interval: Optional[int] = None  # 10
    save_prefix: Optional[str] = None
    num_workers: Optional[int] = None  # 1
    num_threads: Optional[int] = None  # 0
    gaussian: Optional[int] = None  # 0
    patch_size: Optional[int] = None  # 96
    patch_padding: Optional[int] = None  # 48
    device: Optional[int] = None  # -2
    cleanup_output: bool = True
    relion_options: RelionServiceOptions

    @field_validator("model")
    @classmethod
    def saved_models(cls, v):
        if v not in ["unet-3d-10a", "unet-3d-20a", "unet-3d"]:
            raise ValueError("Model must be one of unet-3d-10a, unet-3d-20a, unet-3d")
        return v

    @field_validator("optim")
    @classmethod
    def optimizers(cls, v):
        if v not in ["adam", "adagrad", "sgd"]:
            raise ValueError("Optimizer must be one of adam, adagrad, sgd")
        return v

    @field_validator("criteria")
    @classmethod
    def training_criteria(cls, v):
        if v not in ["L1", "L2"]:
            raise ValueError("Optimizer must be one of L1, L2")
        return v


class Denoise(CommonService):
    """
    A service for denoising cryoEM tomograms using Topaz
    """

    # Human readable service name
    _service_name = "Denoise"

    # Logger name
    _logger_name = "cryoemservices.services.denoise"

    # Job name
    job_type = "relion.denoisetomo"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Denoise service starting")
        workflows.recipe.wrap_subscribe(
            self._transport,
            self._environment["queue"] or "denoise",
            self.denoise,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def run_topaz(
        self,
        topaz_command: List[str],
        alignment_output_dir: Path,
        tomogram_volume: Path,
        denoised_full_path: Path,
    ):
        return subprocess.run(topaz_command, capture_output=True)

    def denoise(self, rw, header: dict, message: dict):
        """Main function which interprets and processes received messages"""
        if not rw:
            self.log.info("Received a simple message")
            if not isinstance(message, dict):
                self.log.error("Rejected invalid simple message")
                self._transport.nack(header)
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
            rw.transport.nack(header)
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
            "even_train_path": "-a",
            "odd_train_path": "-b",
            "n_train": "--N-train",
            "n_test": "--N-test",
            "crop": "-c",
            "base_kernel_width": "--base-kernel-width",
            "optim": "--optim",
            "lr": "--lr",
            "criteria": "--criteria",
            "momentum": "--momentum",
            "batch_size": "--batch-size",
            "num_epochs": "--num-epochs",
            "weight_decay": "-w",
            "save_interval": "--save-interval",
            "num_workers": "--num-workers",
            "save_prefix": "--save-prefix",
            "num_threads": "-j",
            "gaussian": "-g",
            "patch_size": "-s",
            "patch_padding": "-p",
            "device": "-d",
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

        # Run topaz either locally or using Slurm
        self.log.info(f"Input: {denoise_params.volume} Output: {denoised_full_path}")
        result = self.run_topaz(
            topaz_command=command,
            alignment_output_dir=alignment_output_dir,
            tomogram_volume=Path(denoise_params.volume),
            denoised_full_path=denoised_full_path,
        )

        # Send to node creator
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
            rw.transport.nack(header)
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

        # Insert the denoised tomogram into ISPyB
        ispyb_parameters = {
            "ispyb_command": "insert_processed_tomogram",
            "file_path": str(denoised_full_path),
            "processing_type": "Denoised",
        }
        rw.send_to("ispyb_connector", ispyb_parameters)

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

        self.log.info(f"Done denoising for {denoise_params.volume}")
        rw.transport.ack(header)
        return
