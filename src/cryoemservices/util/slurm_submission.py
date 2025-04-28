from __future__ import annotations

import datetime
import logging
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests
import yaml
from pydantic import BaseModel

from cryoemservices.util.config import ServiceConfig, config_from_file

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


class JobParams(BaseModel):
    cpus_per_task: Optional[int] = None
    current_working_directory: Optional[str] = None
    environment: Optional[list] = None
    memory_per_node: Optional[dict] = None
    name: Optional[str] = None
    nodes: Optional[str] = None
    partition: Optional[str] = None
    prefer: Optional[str] = None
    standard_output: Optional[str] = None
    standard_error: Optional[str] = None
    tasks: Optional[int] = None
    time_limit: Optional[dict] = None
    tres_per_job: Optional[str] = None


class JobSubmitResponseMsg(BaseModel):
    job_id: Optional[int] = None
    error_code: Optional[int] = None
    error: Optional[str] = None


class SlurmRestApi:
    def __init__(
        self,
        url: str,
        user_name: str,
        user_token: Path,
        version: str = "v0.0.40",
    ):
        self.url = url
        self.version = version
        self.session = requests.Session()
        self.session.headers["X-SLURM-USER-NAME"] = user_name
        if Path(user_token).is_file():
            with open(user_token, "r") as f:
                self.session.headers["X-SLURM-USER-TOKEN"] = f.read().strip()
        else:
            # We got passed a path, but it isn't a valid one
            raise RuntimeError(f"SLURM: API token file {user_token} does not exist")

    def submit_job(self, script: str, job: JobParams) -> JobSubmitResponseMsg:
        response = self.session.post(
            url=f"{self.url}/slurm/{self.version}/job/submit",
            json={"script": script, "job": job.model_dump(exclude_none=True)},
        )
        response.raise_for_status()
        return JobSubmitResponseMsg(**response.json())

    def get_job_status(self, job_id):
        response = self.session.get(
            url=f"{self.url}/slurm/{self.version}/job/{job_id}",
        )
        response.raise_for_status()
        return response.json()["jobs"][0]["job_state"][0]

    def delete_job(self, job_id):
        response = self.session.delete(
            url=f"{self.url}/slurm/{self.version}/job/{job_id}",
        )
        response.raise_for_status()


class JobSubmissionParameters(BaseModel):
    commands: str
    job_name: str
    cpus_per_task: Optional[int] = None
    environment: Optional[dict[str, str]] = None
    gpus: Optional[int] = None
    memory_per_node: Optional[int] = None
    nodes: Optional[int] = None
    partition: Optional[str] = None
    prefer: Optional[str] = None
    tasks: Optional[int] = None
    time_limit: Optional[datetime.timedelta] = None


def submit_to_slurm(
    params: JobSubmissionParameters,
    working_directory: Path,
    stdout_file: Path,
    stderr_file: Path,
    logger: logging.Logger,
    service_config: ServiceConfig,
    cluster_name: str,
) -> int | None:
    slurm_credentials = service_config.slurm_credentials.get(cluster_name)
    if not slurm_credentials:
        logger.error("No slurm credentials have been provided, aborting")
        return None
    with open(slurm_credentials, "r") as f:
        slurm_rest = yaml.safe_load(f)
    api = SlurmRestApi(
        url=slurm_rest["url"],
        user_name=slurm_rest["user"],
        user_token=slurm_rest["user_token"],
        version=slurm_rest["api_version"],
    )

    script = params.commands
    if "#!/bin/bash" not in script:
        script = f"#!/bin/bash\n. /etc/profile.d/modules.sh\n{script}"

    if params.environment:
        environment = [f"{k}={v}" for k, v in params.environment.items()]
    else:
        # The environment must not be empty
        minimal_environment = {"USER"}
        # Only attempt to copy variables that already exist.
        minimal_environment &= set(os.environ)
        environment = [f"{k}={os.environ[k]}" for k in minimal_environment]

    # Partition and preference always need to match
    if params.partition:
        partition_to_use = params.partition
        preferred_partition = params.prefer
    else:
        partition_to_use = slurm_rest.get("partition")
        preferred_partition = slurm_rest.get("partition_preference")

    logger.info(f"Submitting script to Slurm:\n{script}")
    jdm_params = JobParams(
        cpus_per_task=params.cpus_per_task,
        current_working_directory=str(working_directory),
        standard_output=str(stdout_file),
        standard_error=str(stderr_file),
        environment=environment,
        name=params.job_name,
        nodes=str(params.nodes) if params.nodes else params.nodes,
        partition=partition_to_use,
        prefer=preferred_partition,
        tasks=params.tasks,
    )
    if params.memory_per_node:
        jdm_params.memory_per_node = {
            "number": params.memory_per_node,
            "set": True,
            "infinite": False,
        }
    if params.time_limit:
        time_limit_minutes = math.ceil(params.time_limit.total_seconds() / 60)
        jdm_params.time_limit = {
            "number": time_limit_minutes,
            "set": True,
            "infinite": False,
        }
    if params.gpus:
        jdm_params.tres_per_job = f"gres/gpu:{params.gpus}"

    try:
        response = api.submit_job(script=script, job=jdm_params)
    except Exception as e:
        logger.error(f"Failed Slurm job submission: {e}\n" f"{e}")
        return None
    if response.error:
        error_message = f"{response.error_code}: {response.error}"
        logger.error(f"Failed Slurm job submission: {error_message}")
        return None
    return response.job_id


def wait_for_job_completion(
    job_id: int,
    logger: logging.Logger,
    service_config: ServiceConfig,
    cluster_name: str,
    timeout_counter: int = 60,
) -> str:
    slurm_credentials = service_config.slurm_credentials.get(cluster_name)
    if not slurm_credentials:
        logger.error("No slurm credentials have been provided, aborting")
        return "UNKNOWN"
    with open(slurm_credentials, "r") as f:
        slurm_rest = yaml.safe_load(f)
    api = SlurmRestApi(
        url=slurm_rest["url"],
        user_name=slurm_rest["user"],
        user_token=slurm_rest["user_token"],
        version=slurm_rest["api_version"],
    )

    slurm_job_state = "PENDING"
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
        try:
            slurm_job_state = api.get_job_status(job_id)
        except Exception as e:
            logger.error(f"Failed to get job state: {e}\n" f"{e}")
            return "UNKNOWN"

        if loop_counter >= timeout_counter:
            # Cancel any jobs exceeding a given time threshold
            api.delete_job(job_id)
            logger.error(f"Timeout running job {job_id}")
            return "CANCELLED"
    return slurm_job_state


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


def slurm_submission_for_services(
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
) -> subprocess.CompletedProcess:
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

    # Get the status of the submitted job from the restAPI
    log.info(f"Submitted job {job_id} for {job_name} to slurm. Waiting...")
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
