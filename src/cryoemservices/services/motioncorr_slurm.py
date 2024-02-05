from __future__ import annotations

import json
import os
import string
import subprocess
import time
from collections import ChainMap
from pathlib import Path

import yaml
from workflows.services.common_service import CommonService

from cryoemservices.services.motioncorr import MotionCorr

"""
This service submits motion correction jobs to a slurm cluster
To do this it needs environment variables set for the following:
    MOTIONCOR2_SIF: singularity image of MotionCor2
    SLURM_RESTAPI_CONFIG: configuration yaml file for the slurm cluster

The configuration has the following format:
    plugin: slurm
    url: <url>:<port>
    user_token: <file with restapi token>
    user: <username>
    user_home: <home directory>
    api_version: v0.0.38 or v0.0.40
    partition: <optional slurm partition>
    partition_preference: <optional slurm preferences>
    clusters: <optional slurm clusters>
    required_directories: [<list of directories to bind for singularity>]
"""

slurm_json_job_template = {
    "v0.0.38": {
        "name": "MotionCorr",
        "nodes": 1,
        "tasks": 1,
        "cpus_per_task": 1,
        "gpus": 1,
        "memory_per_gpu": 12000,
        "time_limit": "1:00:00",
    },
    "v0.0.40": {
        "name": "MotionCorr",
        "minimum_nodes": 1,
        "maximum_nodes": 1,
        "tasks": 1,
        "cpus_per_task": 1,
        "tres_per_task": "gres/gpu:1",
        "memory_per_node": {
            "number": 12000,
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
    "echo \"$(date '+%Y-%m-%d %H:%M:%S.%3N'): running MotionCor2\"\n"
    "mkdir /tmp/tmp_$SLURM_JOB_ID\n"
    "export APPTAINER_CACHEDIR=/tmp/tmp_$SLURM_JOB_ID\n"
    "export APPTAINER_TMPDIR=/tmp/tmp_$SLURM_JOB_ID\n"
    "singularity exec --nv --bind /lib64,/tmp/tmp_$SLURM_JOB_ID:/tmp"
)
slurm_tmp_cleanup = "\nrm -rf /tmp/tmp_$SLURM_JOB_ID"


class ChainMapWithReplacement(ChainMap):
    def __init__(self, *maps, substitutions=None) -> None:
        super().__init__(*maps)
        self._substitutions = substitutions

    def __getitem__(self, k):
        v = super().__getitem__(k)
        if self._substitutions and isinstance(v, str) and "$" in v:
            template = string.Template(v)
            return template.substitute(**self._substitutions)
        return v


class MotionCorrSlurm(MotionCorr, CommonService):
    """
    A service for submitting MotionCor2 jobs to slurm via a RestAPI
    """

    # Logger name
    _logger_name = "cryoemservices.services.motioncorr_slurm"

    def parse_mc_slurm_output(self, mc_output_file):
        """
        Read the output logs of MotionCorr to determine
        the movement of each frame
        """
        with open(mc_output_file, "r") as mc_file:
            lines = mc_file.readlines()
            frames_line = False
            for line in lines:
                # Frame reading in MotionCorr 1.4.0
                if line.startswith("...... Frame"):
                    line_split = line.split()
                    self.x_shift_list.append(float(line_split[-2]))
                    self.y_shift_list.append(float(line_split[-1]))

                # Alternative frame reading for MotionCorr 1.6.3
                if not line:
                    frames_line = False
                if frames_line:
                    line_split = line.split()
                    self.x_shift_list.append(float(line_split[1]))
                    self.y_shift_list.append(float(line_split[2]))
                if "x Shift" in line:
                    frames_line = True

    def motioncor2(self, command: list, mrc_out: Path):
        """Submit MotionCor2 jobs to the slurm cluster via the RestAPI"""
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
        if api_version not in ["v0.0.38", "v0.0.40"]:
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout="".encode("utf8"),
                stderr=f"Unsupported API version {api_version}".encode("utf8"),
            )

        # Construct the json for submission
        mc_output_file = f"{mrc_out}.out"
        mc_error_file = f"{mrc_out}.err"
        submission_file = f"{mrc_out}.json"
        slurm_config = {
            "standard_output": mc_output_file,
            "standard_error": mc_error_file,
            "current_working_directory": str(Path(mrc_out).parent),
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
        if slurm_rest.get("required_directories"):
            binding_dirs = "," + ",".join(slurm_rest["required_directories"])
        else:
            binding_dirs = ""
        job_command = (
            slurm_script_template
            + f"{binding_dirs} --home {user_home} {os.environ['MOTIONCOR2_SIF']} "
            + " ".join(command)
            + slurm_tmp_cleanup
        )
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
        self.log.info(f"Submitted MotionCorr job {job_id} to slurm. Waiting...")
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
            except (json.JSONDecodeError, IndexError, KeyError):
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

        # Read in the MotionCor output then clean up the files
        self.log.info(f"Job {job_id} has finished!")
        try:
            self.parse_mc_slurm_output(mc_output_file)
            with open(mc_output_file, "r") as mc_stdout:
                stdout = mc_stdout.read()
            with open(mc_error_file, "r") as mc_stderr:
                stderr = mc_stderr.read()
        except FileNotFoundError:
            self.log.error(f"MotionCor output file {mc_output_file} not found")
            stdout = ""
            stderr = f"Reading MotionCor output file {mc_output_file} failed"
            slurm_job_state = "FAILED"

        if self.x_shift_list and self.y_shift_list:
            Path(mc_output_file).unlink()
            Path(mc_error_file).unlink()
            Path(submission_file).unlink()
        else:
            self.log.error(f"Reading shifts from {mc_output_file} failed")
            slurm_job_state = "FAILED"

        if slurm_job_state == "COMPLETED":
            return subprocess.CompletedProcess(
                args="",
                returncode=0,
                stdout=stdout.encode("utf8"),
                stderr=stderr.encode("utf8"),
            )
        else:
            self.x_shift_list = []
            self.y_shift_list = []
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout=stdout.encode("utf8"),
                stderr=stderr.encode("utf8"),
            )
