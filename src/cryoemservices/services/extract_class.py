from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import workflows.recipe
from pydantic import BaseModel, Field, ValidationError
from workflows.services.common_service import CommonService

from cryoemservices.util.spa_relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)


class ExtractClassParameters(BaseModel):
    micrographs_file: str = Field(..., min_length=1)
    class3d_dir: str = Field(..., min_length=1)
    refine_job_dir: str = Field(..., min_length=1)
    refine_class_nr: int
    boxsize: int
    pixel_size: float
    downscaled_pixel_size: float
    nr_iter_3d: int = 20
    bg_radius: int = -1
    relion_options: RelionServiceOptions


class ExtractClass(CommonService):
    """
    A service for extracting particles from a class for refinement
    """

    # Human readable service name
    _service_name = "ExtractClass"

    # Logger name
    _logger_name = "cryoemservices.services.extract_class"

    # Job name
    select_job_type = "relion.select.onvalue"
    extract_job_type = "relion.extract.reextract"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Class extraction service starting")
        workflows.recipe.wrap_subscribe(
            self._transport,
            "extract_class",
            self.extract_class,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def extract_class(self, rw, header: dict, message: dict):
        class MockRW:
            def dummy(self, *args, **kwargs):
                pass

        if not rw:
            print(
                "Incoming message is not a recipe message. Simple messages can be valid"
            )
            if (
                not isinstance(message, dict)
                or not message.get("parameters")
                or not message.get("content")
            ):
                self.log.error("Rejected invalid simple message")
                self._transport.nack(header)
                return
            self.log.debug("Received a simple message")

            # Create a wrapper-like object that can be passed to functions
            # as if a recipe wrapper was present.
            rw = MockRW()
            rw.transport = self._transport
            rw.recipe_step = {"parameters": message["parameters"]}
            rw.environment = {"has_recipe_wrapper": False}
            rw.set_default_channel = rw.dummy
            rw.send = rw.dummy
            message = message["content"]

        try:
            if isinstance(message, dict):
                extract_params = ExtractClassParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                extract_params = ExtractClassParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Class extraction parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        self.log.info(
            f"Running extraction of particles for {extract_params.class3d_dir} "
            f"class {extract_params.refine_class_nr}"
        )

        # Run in the project directory
        project_dir = Path(extract_params.refine_job_dir).parent.parent
        os.chdir(project_dir)
        self.log.info(
            f"Input: {extract_params.class3d_dir}, Output: {extract_params.refine_job_dir}"
        )
        job_num_refine = int(
            re.search("/job[0-9]{3}", extract_params.refine_job_dir)[0][4:7]
        )

        # Link the required files
        particles_data = (
            Path(extract_params.class3d_dir)
            / f"run_it{extract_params.nr_iter_3d:03}_data.star"
        )

        # Update the relion options
        extract_params.relion_options = update_relion_options(
            extract_params.relion_options, dict(extract_params)
        )

        # Select the particles from the requested class
        select_job_dir = project_dir / f"Select/job{job_num_refine - 2:03}"
        Path(select_job_dir).mkdir(parents=True, exist_ok=True)

        refine_selection_link = Path(
            project_dir / f"Select/Refine_class{extract_params.refine_class_nr}"
        )
        refine_selection_link.unlink(missing_ok=True)
        refine_selection_link.symlink_to(f"job{job_num_refine - 2:03}")

        self.log.info(f"Running {self.select_job_type} in {select_job_dir}")
        select_command = [
            "relion_star_handler",
            "--i",
            str(particles_data),
            "--o",
            f"{select_job_dir}/particles.star",
            "--select",
            "rlnClassNumber",
            "--minval",
            str(extract_params.refine_class_nr),
            "--maxval",
            str(extract_params.refine_class_nr),
            "--pipeline_control",
            f"{select_job_dir}/",
        ]
        select_result = subprocess.run(
            select_command, cwd=str(project_dir), capture_output=True
        )

        # Register the Selection job with the node creator
        self.log.info(f"Sending {self.select_job_type} to node creator")
        node_creator_select = {
            "job_type": self.select_job_type,
            "input_file": str(particles_data),
            "output_file": f"{select_job_dir}/particles.star",
            "relion_options": dict(extract_params.relion_options),
            "command": " ".join(select_command),
            "stdout": select_result.stdout.decode("utf8", "replace"),
            "stderr": select_result.stderr.decode("utf8", "replace"),
        }
        if select_result.returncode:
            node_creator_select["success"] = False
        else:
            node_creator_select["success"] = True
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="node_creator",
                message={"parameters": node_creator_select, "content": "dummy"},
            )
        else:
            rw.send_to("node_creator", node_creator_select)

        # End here if the command failed
        if select_result.returncode:
            self.log.error(
                "Refinement selection failed with exitcode "
                f"{select_result.returncode}:\n"
                + select_result.stderr.decode("utf8", "replace")
            )
            return False

        # Find the number of particles in the class
        number_of_particles = select_result.stdout.decode("utf8", "replace").split()[3]

        # Run re-extraction on the selected particles
        extract_job_dir = project_dir / f"Extract/job{job_num_refine - 1:03}"
        extract_job_dir.mkdir(parents=True, exist_ok=True)

        refine_extraction_link = Path(
            project_dir / f"Extract/Reextract_class{extract_params.refine_class_nr}"
        )
        refine_extraction_link.unlink(missing_ok=True)
        refine_extraction_link.symlink_to(f"job{job_num_refine - 1:03}")

        # If no background radius set diameter as 75% of box
        if extract_params.bg_radius == -1:
            extract_params.bg_radius = round(0.375 * extract_params.boxsize)

        self.log.info(f"Running {self.extract_job_type} in {extract_job_dir}")
        extract_command = [
            "relion_preprocess",
            "--i",
            extract_params.micrographs_file,
            "--reextract_data_star",
            f"{select_job_dir}/particles.star",
            "--recenter",
            "--recenter_x",
            "0",
            "--recenter_y",
            "0",
            "--recenter_z",
            "0",
            "--part_star",
            str(extract_job_dir / "particles.star"),
            "--pick_star",
            str(extract_job_dir / "extractpick.star"),
            "--part_dir",
            str(extract_job_dir),
            "--extract",
            "--extract_size",
            str(extract_params.boxsize),
            "--norm",
            "--bg_radius",
            str(extract_params.bg_radius),
            "--invert_contrast",
            "--pipeline_control",
            f"{extract_job_dir}/",
        ]
        extract_result = subprocess.run(
            extract_command, cwd=str(project_dir), capture_output=True
        )

        # Register the Re-extraction job with the node creator
        self.log.info(f"Sending {self.extract_job_type} to node creator")
        node_creator_extract = {
            "job_type": self.extract_job_type,
            "input_file": f"{select_job_dir}/particles.star:{extract_params.micrographs_file}",
            "output_file": f"{extract_job_dir}/particles.star",
            "relion_options": dict(extract_params.relion_options),
            "command": " ".join(extract_command),
            "stdout": extract_result.stdout.decode("utf8", "replace"),
            "stderr": extract_result.stderr.decode("utf8", "replace"),
        }
        if extract_result.returncode:
            node_creator_extract["success"] = False
        else:
            node_creator_extract["success"] = True
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="node_creator",
                message={"parameters": node_creator_extract, "content": "dummy"},
            )
        else:
            rw.send_to("node_creator", node_creator_extract)

        # End here if the command failed
        if extract_result.returncode:
            self.log.error(
                "Refinement re-extraction failed with exitcode "
                f"{extract_result.returncode}:\n"
                + extract_result.stderr.decode("utf8", "replace")
            )
            return False

        # Create a reference for the refinement
        class_reference = (
            Path(extract_params.class3d_dir)
            / f"run_it{extract_params.nr_iter_3d:03}_class{extract_params.refine_class_nr:03}.mrc"
        )
        rescaled_class_reference = (
            extract_job_dir
            / f"refinement_reference_class{extract_params.refine_class_nr:03}.mrc"
        )

        self.log.info("Running class reference rescaling")
        rescale_command = [
            "relion_image_handler",
            "--i",
            str(class_reference),
            "--o",
            str(rescaled_class_reference),
            "--angpix",
            str(extract_params.downscaled_pixel_size),
            "--rescale_angpix",
            str(extract_params.pixel_size),
            "--new_box",
            str(extract_params.boxsize),
        ]
        rescale_result = subprocess.run(
            rescale_command, cwd=str(project_dir), capture_output=True
        )

        # End here if the command failed
        if rescale_result.returncode:
            self.log.error(
                "Refinement reference scaling failed with exitcode "
                f"{rescale_result.returncode}:\n"
                + rescale_result.stderr.decode("utf8", "replace")
            )
            return False

        # Send on to the refinement wrapper
        refine_params = {
            "refine_job_dir": extract_params.refine_job_dir,
            "particles_file": f"{extract_job_dir}/particles.star",
            "rescaled_class_reference": str(rescaled_class_reference),
            "is_first_refinement": True,
            "number_of_particles": number_of_particles,
            "batch_size": number_of_particles,
            "pixel_size": extract_params.pixel_size,
            "class_number": extract_params.refine_class_nr,
        }
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="refine_wrapper",
                message={"parameters": refine_params, "content": "dummy"},
            )
        else:
            rw.send_to("refine_wrapper", refine_params)

        self.log.info(f"Done {self.extract_job_type} for {extract_params.class3d_dir}.")
        rw.transport.ack(header)
