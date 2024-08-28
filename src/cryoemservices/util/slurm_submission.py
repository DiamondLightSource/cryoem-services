from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

""""
This service submits jobs to a slurm cluster
To do this it needs environment variables set for the following:
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
    cluster: <optional slurm clusters>
    required_directories: [<list of directories to bind for singularity>]
"""

slurm_json_job_template = {
    "v0.0.38": {
        "nodes": 1,
        "tasks": 1,
        "time_limit": "1:00:00",
    },
    "v0.0.40": {
        "minimum_nodes": 1,
        "maximum_nodes": 1,
        "tasks": 1,
        "memory_per_node": {
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

# Templates for running a command using singularity or by with a module/executable
singularity_script_template = (
    "#!/bin/bash\n"
    "echo \"$(date '+%Y-%m-%d %H:%M:%S.%3N'): running slurm job\"\n"
    "mkdir /tmp/tmp_$SLURM_JOB_ID\n"
    "export APPTAINER_CACHEDIR=/tmp/tmp_$SLURM_JOB_ID\n"
    "export APPTAINER_TMPDIR=/tmp/tmp_$SLURM_JOB_ID\n"
)
module_script_template = (
    "#!/bin/bash\n"
    "echo \"$(date '+%Y-%m-%d %H:%M:%S.%3N'): running slurm job\"\n"
    "source /etc/profile.d/modules.sh\n"
)
slurm_tmp_cleanup = "\nrm -rf /tmp/tmp_$SLURM_JOB_ID"


def slurm_submission(
    log,
    job_name: str,
    command: list,
    project_dir: Path,
    output_file: Path,
    cpus: int,
    use_gpu: bool,
    use_singularity: bool,
    cif_name: str = "",
    script_extras: str = "",
    memory_request: int = 12000,
    external_filesystem: bool = False,
):
    """Submit jobs to a slurm cluster via the RestAPI"""
    try:
        # Get the configuration and token for the restAPI
        with open(os.environ["SLURM_RESTAPI_CONFIG"], "r") as f:
            slurm_rest = yaml.safe_load(f)
        user = slurm_rest["user"]
        user_home = slurm_rest["user_home"]
        with open(slurm_rest["user_token"], "r") as f:
            slurm_token = f.read().strip()
    except (KeyError, FileNotFoundError):
        log.error("Unable to load slurm restAPI config file and token")
        return subprocess.CompletedProcess(
            args="",
            returncode=1,
            stdout="".encode("utf8"),
            stderr="No restAPI config or token".encode("utf8"),
        )

    # Check the API version is one this service has been tested with
    api_version = slurm_rest["api_version"]
    if api_version not in ["v0.0.38", "v0.0.40"]:
        return subprocess.CompletedProcess(
            args="",
            returncode=1,
            stdout="".encode("utf8"),
            stderr=f"Unsupported API version {api_version}".encode("utf8"),
        )

    # Construct the json for submission
    slurm_output_file = f"{output_file}.out"
    slurm_error_file = f"{output_file}.err"
    submission_file = f"{output_file}.json"
    slurm_config: dict[str, Any] = {
        "standard_output": slurm_output_file,
        "standard_error": slurm_error_file,
        "current_working_directory": str(project_dir),
    }
    if api_version == "v0.0.38":
        slurm_config["environment"] = {"USER": user, "HOME": user_home}
    else:
        slurm_config["environment"] = [f"USER={user}", f"HOME={user_home}"]

    # Add slurm partition and cluster preferences if given
    if slurm_rest.get("partition"):
        slurm_config["partition"] = slurm_rest["partition"]
    if slurm_rest.get("partition_preference"):
        slurm_config["prefer"] = slurm_rest["partition_preference"]
    if slurm_rest.get("cluster"):
        slurm_config["cluster"] = slurm_rest["cluster"]

    # Combine this with the template for the given API version
    slurm_json_job: dict[str, Any] = dict(
        slurm_json_job_template[api_version], **slurm_config
    )
    slurm_json_job["name"] = job_name
    slurm_json_job["cpus_per_task"] = cpus
    if use_gpu:
        if api_version == "v0.0.38":
            slurm_json_job["gpus"] = 1
            slurm_json_job["memory_per_gpu"] = memory_request
        else:
            slurm_json_job["tres_per_task"] = "gres/gpu:1"
            slurm_json_job["memory_per_node"]["number"] = memory_request
    else:
        if api_version == "v0.0.38":
            slurm_json_job["memory_per_cpu"] = 1000
        else:
            slurm_json_job["memory_per_node"]["number"] = 1000 * cpus

    # Construct the job command and save the job script
    if use_singularity:
        if slurm_rest.get("required_directories"):
            binding_dirs = "," + ",".join(slurm_rest["required_directories"])
        else:
            binding_dirs = ""
        job_command = (
            singularity_script_template
            + script_extras
            + "\n"
            + "singularity exec --nv --bind /lib64,/tmp/tmp_$SLURM_JOB_ID:/tmp"
            + f"{binding_dirs} --home {user_home} {cif_name} "
            + " ".join(command)
            + slurm_tmp_cleanup
        )
    else:
        job_command = module_script_template + script_extras + "\n" + " ".join(command)
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
        log.error(
            f"Unable to submit job to {slurm_rest['url']}. The restAPI returned "
            f"{slurm_submission_json.stdout.decode('utf8', 'replace')}"
        )
        return subprocess.CompletedProcess(
            args="",
            returncode=1,
            stdout=slurm_submission_json.stdout,
            stderr=slurm_submission_json.stderr,
        )
    log.info(f"Submitted job {job_id} for {job_name} to slurm. Waiting...")
    if slurm_response_json.get("warnings") and slurm_response_json["warnings"]:
        log.warning(f"Slurm reported these warnings: {slurm_response_json['warnings']}")
    if slurm_response_json.get("errors") and slurm_response_json["errors"]:
        log.warning(f"Slurm reported these errors: {slurm_response_json['errors']}")

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
            log.error(
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
            log.error(f"Timeout running {job_name}")
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout="".encode("utf8"),
                stderr=f"Timeout running {job_name}".encode("utf8"),
            )

    # Read in the output
    log.info(f"Job {job_id} has finished!")
    if not external_filesystem:
        try:
            with open(slurm_output_file, "r") as slurm_stdout:
                stdout = slurm_stdout.read()
            with open(slurm_error_file, "r") as slurm_stderr:
                stderr = slurm_stderr.read()
        except FileNotFoundError:
            log.error(f"Output file {slurm_output_file} not found")
            stdout = ""
            stderr = f"Reading output file {slurm_error_file} failed"
            slurm_job_state = "FAILED"
    else:
        stdout = ""
        stderr = ""

    # Read in the output then clean up the files
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
