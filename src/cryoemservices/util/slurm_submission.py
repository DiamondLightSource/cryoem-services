from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import yaml

from cryoemservices.services.cluster_submission import (
    JobSubmissionParameters,
    submit_to_slurm,
    wait_for_job_completion,
)
from cryoemservices.util.config import config_from_file

""""
This service submits jobs to a slurm cluster

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
    service_config_file: Path,
    slurm_cluster: str,
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
    extra_singularity_directories: Optional[list[str]] = None,
):
    """Submit jobs to a slurm cluster via the RestAPI"""
    # Load the service config with slurm credentials
    service_config = config_from_file(service_config_file)
    slurm_credentials = service_config.slurm_credentials.get(slurm_cluster)
    if not slurm_credentials:
        log.error("No slurm credentials have been provided, aborting")
        return subprocess.CompletedProcess(
            args="",
            returncode=1,
            stdout="".encode("utf8"),
            stderr="No slurm credentials found".encode("utf8"),
        )

    try:
        # Get the configuration and token for the restAPI
        with open(slurm_credentials, "r") as f:
            slurm_rest = yaml.safe_load(f)
        user = slurm_rest["user"]
        user_home = slurm_rest["user_home"]
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
    if api_version not in ["v0.0.40"]:
        return subprocess.CompletedProcess(
            args="",
            returncode=1,
            stdout="".encode("utf8"),
            stderr=f"Unsupported API version {api_version}".encode("utf8"),
        )

    # Output log files
    slurm_output_file = output_file.with_suffix(".out")
    slurm_error_file = output_file.with_suffix(".err")

    # Construct the job command and save the job script
    if use_singularity:
        if slurm_rest.get("required_directories"):
            binding_dirs = "," + ",".join(slurm_rest["required_directories"])
        else:
            binding_dirs = ""
        if extra_singularity_directories:
            for extra_binding_dir in extra_singularity_directories:
                binding_dirs += f",{extra_binding_dir}"
        job_command = (
            singularity_script_template
            + script_extras
            + "\n"
            + "singularity exec --nv --bind /tmp/tmp_$SLURM_JOB_ID:/tmp"
            + f"{binding_dirs} --home {user_home} {cif_name} "
            + " ".join(command)
            + slurm_tmp_cleanup
        )
    else:
        job_command = module_script_template + script_extras + "\n" + " ".join(command)

    # Command to submit jobs to the restAPI
    job_params = JobSubmissionParameters(
        job_name=job_name,
        environment={"USER": user, "HOME": user_home},
        cpus_per_task=cpus,
        tasks=1,
        nodes=1,
        memory_per_node=memory_request if use_gpu else 1000 * cpus,
        time_limit=3600,
        gpus=1 if use_gpu else None,
        commands=job_command,
    )

    job_id = submit_to_slurm(
        params=job_params,
        working_directory=project_dir,
        stdout_file=slurm_output_file,
        stderr_file=slurm_error_file,
        logger=log,
        service_config=service_config,
        cluster_name=slurm_cluster,
    )
    if not job_id:
        log.error(f"Unable to submit job to {slurm_rest['url']}")
        return subprocess.CompletedProcess(
            args="",
            returncode=1,
            stdout="cluster job submission".encode("utf8"),
            stderr="failed to submit job".encode("utf8"),
        )
    log.info(f"Submitted job {job_id} for {job_name} to slurm. Waiting...")

    # Command to get the status of the submitted job from the restAPI
    slurm_job_state = wait_for_job_completion(
        job_id=job_id,
        logger=log,
        service_config=service_config,
        cluster_name=slurm_cluster,
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
