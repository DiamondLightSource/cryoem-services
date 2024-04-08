from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import workflows.recipe
import yaml
from pydantic import BaseModel, Field, ValidationError
from workflows.services.common_service import CommonService

from cryoemservices.util.spa_relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)


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


slurm_json_job_template = {
    "v0.0.38": {
        "name": "ReExtract",
        "nodes": 1,
        "tasks": 1,
        "cpus_per_task": 40,
        "memory_per_cpu": 7000,
        "time_limit": "1:00:00",
    },
    "v0.0.40": {
        "name": "ReExtract",
        "minimum_nodes": 1,
        "maximum_nodes": 1,
        "tasks": 1,
        "cpus_per_task": 40,
        "memory_per_cpu": {
            "number": 7000,
            "set": True,
            "infinite": False,
        },
        "time_limit": {
            "number": 3600,
            "set": True,
            "infinite": False,
        },
    },
}
slurm_script_template = (
    "#!/bin/bash\necho \"$(date '+%Y-%m-%d %H:%M:%S.%3N'): running ReExtraction\"\n"
    "source /etc/profile.d/modules.sh\n"
    "module load EM/cryoem-services\n"
)


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

    def submit_slurm_job(self, command, job_dir):
        """Submit ReExtract jobs to the slurm cluster via the RestAPI"""
        try:
            # Get the configuration and token for the restAPI
            with open(os.environ["SLURM_RESTAPI_CONFIG"], "r") as f:
                slurm_rest = yaml.safe_load(f)
            user = slurm_rest["user"]
            user_home = slurm_rest["user_home"]
            with open(slurm_rest["user_token"], "r") as f:
                slurm_token = f.read().strip()
        except (KeyError, FileNotFoundError):
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout="".encode("utf8"),
                stderr="No restAPI config or token".encode("utf8"),
            )

        # Check the API version is one this service has been tested with
        api_version = slurm_rest["api_version"]
        print(api_version)
        if api_version not in ["v0.0.38", "v0.0.40"]:
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout="".encode("utf8"),
                stderr=f"Unsupported API version {api_version}".encode("utf8"),
            )

        # Construct the json for submission
        slurm_output_file = f"{job_dir}/slurm_run.out"
        slurm_error_file = f"{job_dir}/slurm_run.err"
        submission_file = f"{job_dir}/slurm_run.json"
        slurm_config = {
            "standard_output": slurm_output_file,
            "standard_error": slurm_error_file,
            "current_working_directory": str(job_dir),
        }
        if api_version == "v0.0.38":
            slurm_config["environment"] = {"USER": user, "HOME": user_home}
        else:
            slurm_config["environment"] = [f"USER: {user}", f"HOME: {user_home}"]

        # Add slurm partition and cluster preferences if given
        if slurm_rest.get("partition"):
            slurm_config["partition"] = slurm_rest["partition"]
        if slurm_rest.get("partition_preference"):
            slurm_config["prefer"] = slurm_rest["partition_preference"]
        if slurm_rest.get("clusters"):
            slurm_config["clusters"] = slurm_rest["clusters"]
        # Combine this with the template for the given API version
        slurm_json_job = dict(slurm_json_job_template[api_version], **slurm_config)

        # Make the script command and save the submission json
        job_command = slurm_script_template + " ".join(command)
        slurm_json = {"job": slurm_json_job, "script": job_command}
        with open(submission_file, "w") as f:
            json.dump(slurm_json, f)

        # Command to submit jobs to the restAPI
        slurm_submit_command = (
            f'curl -H "X-SLURM-USER-NAME:{user}" -H "X-SLURM-USER-TOKEN:{slurm_token}" '
            '-H "Content-Type: application/json" -X POST '
            f'{slurm_rest["url"]}/slurm/{slurm_rest["api_version"]}/job/submit '
            f"-d @{submission_file}"
        )
        slurm_submission_json = subprocess.run(
            slurm_submit_command, capture_output=True, shell=True
        )
        try:
            # Extract the job id from the submission response to use in the next query
            slurm_response = slurm_submission_json.stdout.decode("utf8", "replace")
            slurm_response_json = json.loads(slurm_response)
            job_id = slurm_response_json["job_id"]
        except (json.JSONDecodeError, KeyError):
            self.log.error(
                f"Unable to submit job to {slurm_rest['url']}. The restAPI returned "
                f"{slurm_submission_json.stdout.decode('utf8', 'replace')}"
            )
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout=slurm_submission_json.stdout,
                stderr=slurm_submission_json.stderr,
            )
        self.log.info(f"Submitted job {job_id} to slurm. Waiting...")
        if slurm_response_json.get("warnings") and slurm_response_json["warnings"]:
            self.log.warning(
                f"Slurm reported these warnings: {slurm_response_json['warnings']}"
            )
        if slurm_response_json.get("errors") and slurm_response_json["errors"]:
            self.log.warning(
                f"Slurm reported these errors: {slurm_response_json['errors']}"
            )

        # Command to get the status of the submitted job from the restAPI
        slurm_status_command = (
            f'curl -H "X-SLURM-USER-NAME:{user}" -H "X-SLURM-USER-TOKEN:{slurm_token}" '
            '-H "Content-Type: application/json" -X GET '
            f'{slurm_rest["url"]}/slurm/{slurm_rest["api_version"]}/job/{job_id}'
        )
        slurm_job_state = "PENDING"

        # Wait until the job has a status indicating it has finished
        loop_counter = 0
        while slurm_job_state in (
            "PENDING",
            "CONFIGURING",
            "RUNNING",
            "COMPLETING",
        ):
            if loop_counter < 5:
                time.sleep(5)
            else:
                time.sleep(30)
            loop_counter += 1

            # Call the restAPI to find out the job state
            slurm_status_json = subprocess.run(
                slurm_status_command, capture_output=True, shell=True
            )
            try:
                slurm_response = slurm_status_json.stdout.decode("utf8", "replace")
                slurm_job_state = json.loads(slurm_response)["jobs"][0]["job_state"]
                if api_version == "v0.0.40":
                    slurm_job_state = slurm_job_state[0]
            except (json.JSONDecodeError, KeyError):
                print(slurm_status_command)
                self.log.error(
                    f"Unable to get status for job {job_id}. The restAPI returned "
                    f"{slurm_status_json.stdout.decode('utf8', 'replace')}"
                )
                return subprocess.CompletedProcess(
                    args="",
                    returncode=1,
                    stdout=slurm_status_json.stdout,
                    stderr=slurm_status_json.stderr,
                )

            if loop_counter >= 60:
                slurm_cancel_command = (
                    f'curl -H "X-SLURM-USER-NAME:{user}" '
                    f'-H "X-SLURM-USER-TOKEN:{slurm_token}" '
                    '-H "Content-Type: application/json" -X DELETE '
                    f'{slurm_rest["url"]}/slurm/{slurm_rest["api_version"]}/job/{job_id}'
                )
                subprocess.run(slurm_cancel_command, capture_output=True, shell=True)
                self.log.error("Timeout running slurm job")
                return subprocess.CompletedProcess(
                    args="",
                    returncode=1,
                    stdout="".encode("utf8"),
                    stderr="Timeout running slurm job".encode("utf8"),
                )

        # Read in the output then clean up the files
        self.log.info(f"Job {job_id} has finished!")
        try:
            with open(slurm_output_file, "r") as slurm_stdout:
                stdout = slurm_stdout.read()
            with open(slurm_error_file, "r") as slurm_stderr:
                stderr = slurm_stderr.read()
        except FileNotFoundError:
            self.log.error(f"Output file {slurm_output_file} not found")
            stdout = ""
            stderr = f"Reading output file {slurm_output_file} failed"
            slurm_job_state = "FAILED"

        if slurm_job_state == "COMPLETED":
            return subprocess.CompletedProcess(
                args="",
                returncode=0,
                stdout=stdout.encode("utf8"),
                stderr=stderr.encode("utf8"),
            )
        else:
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout=stdout.encode("utf8"),
                stderr=stderr.encode("utf8"),
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

        result = self.submit_slurm_job(command, extract_job_dir)

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
