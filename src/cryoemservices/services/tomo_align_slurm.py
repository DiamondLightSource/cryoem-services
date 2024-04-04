from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List

import datasyncer
import yaml
from requests import HTTPError
from workflows.services.common_service import CommonService

from cryoemservices.services.tomo_align import TomoAlign

"""
This service submits AreTomo jobs to a slurm cluster (currently the STFC IRIS cluster)
To do this it needs environment variables set for the following:
    ARETOMO_SIF: singularity image of AreTomo
    SLURM_RESTAPI_CONFIG: configuration yaml file for the slurm cluster

The configuration has the following format:
    plugin: slurm
    url: <url>:<port>
    user_token: <file with restapi token>
    user: <username>
    user_home: <home directory>
    api_version: v0.0.40
    partition: <optional slurm partition>
    partition_preference: <optional slurm preferences>
    cluster: <optional slurm clusters>
    required_directories: [<list of directories to bind for singularity>]
"""


def retrieve_files(job_directory: Path, files_to_skip: List[Path]):
    """Copy files back from the Iris cluster"""
    iris_directory = Path("/iris") / Path(job_directory).relative_to("/dls")
    for iris_file in iris_directory.glob("*"):
        dls_file = job_directory / iris_file.relative_to(iris_directory)
        if dls_file not in files_to_skip:
            dls_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(iris_file, dls_file)
        iris_file.unlink()


def transfer_files(file_list: List[str]):
    """Transfer files to the Iris cluster"""
    try:
        transfer_id = datasyncer.transfer(file_list)
        status = "active"
        while status == "active":
            time.sleep(5)
            status = datasyncer.status(transfer_id)
    except HTTPError as e:
        return f"Unable to transfer data: {e}"

    if status == "succeeded":
        return 0
    elif status == "failed":
        return "Data syncer reported a failure"
    else:
        return f"Unknown status {status}"


slurm_json_job_template = {
    "v0.0.40": {
        "name": "TomoAlign",
        "minimum_nodes": 1,
        "maximum_nodes": 1,
        "tasks": 1,
        "cpus_per_task": 1,
        "tres_per_task": "gres/gpu:1",
        "memory_per_node": {
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
    "#!/bin/bash\n"
    "echo \"$(date '+%Y-%m-%d %H:%M:%S.%3N'): running AreTomo\"\n"
    "mkdir /tmp/tmp_$SLURM_JOB_ID\n"
    "export APPTAINER_CACHEDIR=/tmp/tmp_$SLURM_JOB_ID\n"
    "export APPTAINER_TMPDIR=/tmp/tmp_$SLURM_JOB_ID\n"
    "singularity exec --nv --bind /tmp/tmp_$SLURM_JOB_ID:/tmp"
)
slurm_tmp_cleanup = "\nrm -rf /tmp/tmp_$SLURM_JOB_ID"


class TomoAlignSlurm(TomoAlign, CommonService):
    """
    A service for submitting AreTomo jobs to a slurm cluster via RestAPI
    """

    # Logger name
    _logger_name = "cryoemservices.services.tomo_align_slurm"

    def parse_tomo_output(self, tomo_output_file):
        tomo_file = open(tomo_output_file, "r")
        lines = tomo_file.readlines()
        for line in lines:
            if line.startswith("Rot center Z"):
                self.rot_centre_z_list.append(line.split()[5])
            if line.startswith("Tilt offset"):
                self.tilt_offset = float(line.split()[2].strip(","))
            if line.startswith("Best tilt axis"):
                self.alignment_quality = float(line.split()[5])
        tomo_file.close()

    def aretomo(self, tomo_parameters):
        """Submit AreTomo jobs to the slurm cluster via the RestAPI"""
        try:
            # Get the configuration and token for the restAPI
            with open(os.environ["SLURM_RESTAPI_CONFIG"], "r") as f:
                slurm_rest = yaml.safe_load(f)
            user = slurm_rest["user"]
            user_home = slurm_rest["user_home"]
            with open(slurm_rest["user_token"], "r") as f:
                slurm_token = f.read().strip()
        except (KeyError, FileNotFoundError):
            self.log.error("Unable to load slurm restAPI config file and token")
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout="".encode("utf8"),
                stderr="No restAPI config or token".encode("utf8"),
            )

        # Check the API version is one this service has been tested with
        api_version = slurm_rest["api_version"]
        print(api_version)
        if api_version not in ["v0.0.40"]:
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout="".encode("utf8"),
                stderr=f"Unsupported API version {api_version}".encode("utf8"),
            )

        # Construct the json for submission
        slurm_output_file = f"{tomo_parameters.aretomo_output_file}.out"
        slurm_error_file = f"{tomo_parameters.aretomo_output_file}.err"
        submission_file = f"{tomo_parameters.aretomo_output_file}.json"
        slurm_config = {
            "environment": [f"USER: {user}", f"HOME: {user_home}"],
            "standard_output": slurm_output_file,
            "standard_error": slurm_error_file,
            "current_working_directory": str(
                Path(tomo_parameters.aretomo_output_file).parent
            ),
        }

        # Add slurm partition and cluster preferences if given
        if slurm_rest.get("partition"):
            slurm_config["partition"] = slurm_rest["partition"]
        if slurm_rest.get("partition_preference"):
            slurm_config["prefer"] = slurm_rest["partition_preference"]
        if slurm_rest.get("cluster"):
            slurm_config["cluster"] = slurm_rest["cluster"]
        slurm_json_job = dict(slurm_json_job_template[api_version], **slurm_config)

        # Copy the AreTomo executable into the visit directory
        aretomo_sif = str(Path(tomo_parameters.stack_file).parent / "AreTomo")
        shutil.copy(os.environ["ARETOMO_SIF"], aretomo_sif)

        # Assemble the command to run AreTomo
        command = [
            aretomo_sif,
            "-OutMrc",
            tomo_parameters.aretomo_output_file,
            "-InMrc",
            str(Path(tomo_parameters.stack_file).name),
        ]

        if tomo_parameters.angle_file:
            command.extend(("-AngFile", tomo_parameters.angle_file))
        else:
            command.extend(
                (
                    "-TiltRange",
                    tomo_parameters.input_file_list[0][1],  # lowest tilt
                    tomo_parameters.input_file_list[-1][1],
                )
            )  # highest tilt

        if tomo_parameters.manual_tilt_offset:
            command.extend(
                (
                    "-TiltCor",
                    str(tomo_parameters.tilt_cor),
                    str(tomo_parameters.manual_tilt_offset),
                )
            )
        elif tomo_parameters.tilt_cor:
            command.extend(("-TiltCor", str(tomo_parameters.tilt_cor)))

        aretomo_flags = {
            "vol_z": "-VolZ",
            "out_bin": "-OutBin",
            "tilt_axis": "-TiltAxis",
            "flip_int": "-FlipInt",
            "flip_vol": "-FlipVol",
            "wbp": "-Wbp",
            "align": "-Align",
            "roi_file": "-RoiFile",
            "patch": "-Patch",
            "kv": "-Kv",
            "align_file": "-AlnFile",
            "align_z": "-AlignZ",
            "pix_size": "-PixSize",
            "init_val": "initVal",
            "refine_flag": "refineFlag",
            "out_imod": "-OutImod",
            "out_imod_xf": "-OutXf",
            "dark_tol": "-DarkTol",
        }

        for k, v in tomo_parameters.dict().items():
            if v and (k in aretomo_flags):
                command.extend((aretomo_flags[k], str(v)))

        # Construct the job command and save the job script
        if slurm_rest.get("required_directories"):
            binding_dirs = "," + ",".join(slurm_rest["required_directories"])
        else:
            binding_dirs = ""
        job_command = (
            slurm_script_template
            + f"{binding_dirs} --home {user_home} "
            + " ".join(command)
            + slurm_tmp_cleanup
        )
        slurm_json = {"job": slurm_json_job, "script": job_command}
        with open(submission_file, "w") as f:
            json.dump(slurm_json, f)

        self.log.info(f"Running AreTomo with command: {command}")
        self.log.info(
            f"Input stack: {tomo_parameters.stack_file} \n"
            f"Output file: {tomo_parameters.aretomo_output_file}"
        )

        # Transfer the required files
        transfer_status = transfer_files([tomo_parameters.stack_file, aretomo_sif])
        if transfer_status:
            self.log.error(f"Unable to transfer files: {transfer_status}")
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout="",
                stderr=transfer_status,
            )

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
                slurm_job_state = json.loads(slurm_response)["jobs"][0]["job_state"][0]
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
                self.log.error("Timeout running motion correction")
                return subprocess.CompletedProcess(
                    args="",
                    returncode=1,
                    stdout="".encode("utf8"),
                    stderr="Timeout running motion correction".encode("utf8"),
                )

        # Get back the output files
        retrieve_files(
            job_directory=Path(tomo_parameters.aretomo_output_file).parent,
            files_to_skip=[tomo_parameters.stack_file, aretomo_sif],
        )
        Path(aretomo_sif).unlink()

        # Read in the output
        self.log.info(f"Job {job_id} has finished!")
        try:
            if tomo_parameters.tilt_cor:
                self.parse_tomo_output(slurm_output_file)
            with open(slurm_output_file, "r") as mc_stdout:
                stdout = mc_stdout.read()
            with open(slurm_error_file, "r") as mc_stderr:
                stderr = mc_stderr.read()
        except FileNotFoundError:
            self.log.error(f"Output file {slurm_output_file} not found")
            stdout = ""
            stderr = f"Reading output file {slurm_error_file} failed"
            slurm_job_state = "FAILED"

        # Read in the output then clean up the files
        self.log.info(f"Job {job_id} has finished!")
        if slurm_job_state == "COMPLETED":
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
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
