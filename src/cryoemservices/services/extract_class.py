from __future__ import annotations

import math
import os
import re
from pathlib import Path

import workflows.recipe
from pydantic import BaseModel, Field, ValidationError
from workflows.services.common_service import CommonService

from cryoemservices.util.relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)
from cryoemservices.util.slurm_submission import slurm_submission


class ExtractClassParameters(BaseModel):
    extraction_executable: str = "cryoemservices.reextract"
    class3d_dir: str = Field(..., min_length=1)
    refine_job_dir: str = Field(..., min_length=1)
    refine_class_nr: int
    original_pixel_size: float
    boxsize: int = 150
    nr_iter_3d: int = 25
    bg_radius: int = -1
    downscale_factor: float = 2
    downscale: bool = True
    normalise: bool = True
    invert_contrast: bool = True
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
    extract_job_type = "relion.extract"

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
            re.search("/job[0-9]+", extract_params.refine_job_dir)[0][4:7]
        )
        original_dir = Path(extract_params.class3d_dir).parent.parent
        ctf_micrographs_file = list(
            project_dir.glob("CtfFind/job00*/micrographs_ctf.star")
        )[0].relative_to(project_dir)

        # Link the required files and pull out necessary parameters
        particles_data = (
            Path(extract_params.class3d_dir)
            / f"run_it{extract_params.nr_iter_3d:03}_data.star"
        )
        with open(
            Path(extract_params.class3d_dir)
            / f"run_it{extract_params.nr_iter_3d:03}_model.star"
        ) as class3d_output:
            postprocess_lines = class3d_output.readlines()
            for line in postprocess_lines:
                if "_rlnPixelSize" in line:
                    downscaled_pixel_size = float(line.split()[-1])
                    break
        if not downscaled_pixel_size:
            self.log.warning("No class3d pixel size found")
            rw.transport.nack(header)
            return

        with open(
            Path(extract_params.class3d_dir)
            / f"run_it{extract_params.nr_iter_3d:03}_optimiser.star"
        ) as class3d_output:
            postprocess_lines = class3d_output.readlines()
            for line in postprocess_lines:
                if "_rlnParticleDiameter" in line:
                    mask_diameter = float(line.split()[-1])
                    break
        if not mask_diameter:
            self.log.warning("No mask diameter found")
            rw.transport.nack(header)
            return

        # Boxsize conversion as in particle extraction, enlarged by 25%
        exact_boxsize = int(
            mask_diameter / extract_params.original_pixel_size / 1.1 * 1.25
        )
        int_boxsize = int(math.ceil(exact_boxsize))
        extract_params.boxsize = int_boxsize + int_boxsize % 2

        # Update the relion options
        extract_params.relion_options = update_relion_options(
            extract_params.relion_options, dict(extract_params)
        )
        if extract_params.downscale:
            exact_scaled_boxsize = (
                extract_params.boxsize / extract_params.downscale_factor
            )
            int_scaled_boxsize = int(math.ceil(exact_scaled_boxsize))
            scaled_pixel_size = (
                extract_params.original_pixel_size * extract_params.downscale_factor
            )
        else:
            int_scaled_boxsize = int(math.ceil(extract_params.boxsize))
            scaled_pixel_size = extract_params.original_pixel_size
        scaled_boxsize = int_scaled_boxsize + int_scaled_boxsize % 2
        extract_params.relion_options.small_boxsize = scaled_boxsize
        extract_params.relion_options.pixel_size_downscaled = scaled_pixel_size

        select_job_dir = project_dir / f"Select/job{job_num_refine - 2:03}"
        extract_job_dir = project_dir / f"Extract/job{job_num_refine - 1:03}"

        # Select the particles from the requested class
        Path(select_job_dir).mkdir(parents=True, exist_ok=True)

        refine_selection_link = Path(
            project_dir / f"Select/Refine_class{extract_params.refine_class_nr}"
        )
        refine_selection_link.unlink(missing_ok=True)
        refine_selection_link.symlink_to(f"job{job_num_refine - 2:03}")

        self.log.info(f"Running {self.select_job_type} in {select_job_dir}")
        number_of_particles = 0
        class_number_row = 19  # usual location of Relion class number loop
        with open(particles_data, "r") as classified_particles, open(
            f"{select_job_dir}/particles.star", "w"
        ) as selected_particles:
            while True:
                line = classified_particles.readline()
                if not line:
                    break
                if line.startswith("_rlnClassNumber"):
                    class_number_row = int(line.split("#")[1]) - 1
                if line.lstrip() and line.lstrip()[0].isnumeric():
                    split_line = line.split()
                    class_number = int(split_line[class_number_row])
                    if class_number != extract_params.refine_class_nr:
                        continue
                    number_of_particles += 1
                selected_particles.write(line)

        # Register the Selection job with the node creator
        self.log.info(f"Sending {self.select_job_type} to node creator")
        node_creator_select = {
            "job_type": self.select_job_type,
            "input_file": str(particles_data),
            "output_file": f"{select_job_dir}/particles.star",
            "relion_options": dict(extract_params.relion_options),
            "command": "",
            "stdout": "",
            "stderr": "",
            "success": True,
        }
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="node_creator",
                message={"parameters": node_creator_select, "content": "dummy"},
            )
        else:
            rw.send_to("node_creator", node_creator_select)

        # Run re-extraction on the selected particles
        extract_job_dir.mkdir(parents=True, exist_ok=True)
        self.log.info(f"Running {self.extract_job_type} in {extract_job_dir}")

        refine_extraction_link = Path(
            project_dir / f"Extract/Reextract_class{extract_params.refine_class_nr}"
        )
        refine_extraction_link.unlink(missing_ok=True)
        refine_extraction_link.symlink_to(f"job{job_num_refine - 1:03}")

        # Make the command, needs the path to the cryoemservices.reextract executable
        command = [
            extract_params.extraction_executable,
            "--extract_job_dir",
            str(extract_job_dir),
            "--select_job_dir",
            str(select_job_dir),
            "--original_dir",
            str(original_dir),
            "--full_boxsize",
            str(extract_params.boxsize),
            "--scaled_boxsize",
            str(scaled_boxsize),
            "--full_pixel_size",
            str(extract_params.original_pixel_size),
            "--scaled_pixel_size",
            str(scaled_pixel_size),
            "--bg_radius",
            str(extract_params.bg_radius),
        ]
        if extract_params.invert_contrast:
            command.append("--invert_contrast")
        if extract_params.normalise:
            command.append("--normalise")
        if extract_params.downscale:
            command.append("--downscale")

        result = slurm_submission(
            log=self.log,
            job_name="ReExtract",
            command=command,
            project_dir=extract_job_dir,
            output_file=extract_job_dir / "slurm_run",
            cpus=40,
            use_gpu=False,
            use_singularity=False,
            script_extras="module load EM/cryoem-services",
        )

        # Register the Re-extraction job with the node creator
        self.log.info(f"Sending {self.extract_job_type} to node creator")
        node_creator_extract = {
            "job_type": self.extract_job_type,
            "input_file": f"{select_job_dir}/particles.star:{ctf_micrographs_file}",
            "output_file": f"{extract_job_dir}/particles.star",
            "relion_options": dict(extract_params.relion_options),
            "command": " ".join(command),
            "stdout": result.stdout.decode("utf8", "replace"),
            "stderr": result.stderr.decode("utf8", "replace"),
        }
        if result.returncode:
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
        if result.returncode:
            self.log.warning(
                f"Reextraction failed: {result.stderr.decode('utf8', 'replace')}"
            )
            rw.transport.nack(header)
            return

        # Create a reference for the refinement
        class_reference = (
            Path(extract_params.class3d_dir)
            / f"run_it{extract_params.nr_iter_3d:03}_class{extract_params.refine_class_nr:03}.mrc"
        )
        rescaled_class_reference = (
            extract_job_dir
            / f"refinement_reference_class{extract_params.refine_class_nr:03}.mrc"
        )

        # Make the scaling command but don't run it here as we don't have Relion
        self.log.info("Setting up class reference rescaling command")
        rescaling_command = [
            "relion_image_handler",
            "--i",
            str(class_reference),
            "--o",
            str(rescaled_class_reference),
            "--angpix",
            str(downscaled_pixel_size),
            "--rescale_angpix",
            str(scaled_pixel_size),
            "--force_header_angpix",
            str(scaled_pixel_size),
            "--new_box",
            str(scaled_boxsize),
        ]

        # Send on to the refinement wrapper
        refine_params = {
            "refine_job_dir": extract_params.refine_job_dir,
            "particles_file": f"{extract_job_dir}/particles.star",
            "rescaling_command": rescaling_command,
            "rescaled_class_reference": str(rescaled_class_reference),
            "is_first_refinement": True,
            "number_of_particles": number_of_particles,
            "batch_size": number_of_particles,
            "pixel_size": str(scaled_pixel_size),
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
