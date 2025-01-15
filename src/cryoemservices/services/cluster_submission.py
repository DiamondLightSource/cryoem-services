from __future__ import annotations

import datetime
import json
import logging
import math
import os
from importlib.metadata import entry_points
from pathlib import Path
from typing import Optional

import requests
import workflows.recipe
import yaml
from pydantic import BaseModel, Field
from workflows.services.common_service import CommonService
from zocalo.util import slurm

from cryoemservices.util.config import ServiceConfig, config_from_file


class JobSubmissionParameters(BaseModel):
    scheduler: str = "slurm"
    job_name: str
    partition: str
    prefer: Optional[str] = None
    environment: Optional[dict[str, str]] = None
    cpus_per_task: Optional[int] = None
    tasks: Optional[int] = None
    nodes: Optional[int] = None
    memory_per_node: Optional[int] = None
    gpus_per_node: Optional[int] = None
    min_memory_per_cpu: Optional[int] = Field(
        None, description="Minimum real memory per cpu (MB)"
    )
    time_limit: Optional[datetime.timedelta] = None
    gpus: Optional[int] = None
    exclusive: bool = False
    commands: list[str] | str


def submit_to_slurm(
    params: JobSubmissionParameters,
    working_directory: Path,
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
    api = slurm.SlurmRestApi(
        url=slurm_rest["url"],
        version=slurm_rest["api_version"],
        user_name=slurm_rest["user"],
        user_token=slurm_rest["user_token"],
    )

    script = params.commands
    if not isinstance(script, str):
        script = "\n".join(script)
    script = f"#!/bin/bash\n. /etc/profile.d/modules.sh\n{script}"

    if params.environment:
        environment = [f"{k}={v}" for k, v in params.environment.items()]
    else:
        # The environment must not be empty
        minimal_environment = {"USER"}
        # Only attempt to copy variables that already exist.
        minimal_environment &= set(os.environ)
        environment = [f"{k}={os.environ[k]}" for k in minimal_environment]
    if not environment:
        logger.error("No environment has been set, aborting")
        return None

    logger.info(f"Submitting script to Slurm:\n{script}")
    jdm_params = {
        "cpus_per_task": params.cpus_per_task,
        "current_working_directory": os.fspath(working_directory),
        "environment": environment,
        "name": params.job_name,
        "nodes": str(params.nodes) if params.nodes else params.nodes,
        "partition": params.partition,
        "prefer": params.prefer,
        "tasks": params.tasks,
    }
    if params.min_memory_per_cpu:
        jdm_params["memory_per_cpu"] = slurm.models.Uint64NoVal(
            number=params.min_memory_per_cpu, set=True
        )
    if params.memory_per_node:
        jdm_params["memory_per_node"] = slurm.models.Uint64NoVal(
            number=params.memory_per_node, set=True
        )
    if params.time_limit:
        time_limit_minutes = math.ceil(params.time_limit.total_seconds() / 60)
        jdm_params["time_limit"] = slurm.models.Uint32NoVal(
            number=time_limit_minutes, set=True
        )
    if params.gpus_per_node:
        jdm_params["tres_per_node"] = f"gres/gpu:{params.gpus_per_node}"
    if params.gpus:
        jdm_params["tres_per_job"] = f"gres/gpu:{params.gpus}"

    job_submission = slurm.models.JobSubmitReq(
        script=script, job=slurm.models.JobDescMsg(**jdm_params)
    )
    try:
        response = api.submit_job(job_submission)
    except requests.HTTPError as e:
        logger.error(f"Failed Slurm job submission: {e}\n" f"{e.response.text}")
        return None
    if response.error:
        error_message = f"{response.error_code}: {response.error}"
        logger.error(f"Failed Slurm job submission: {error_message}")
        return None
    return response.job_id


class ClusterSubmission(CommonService):
    """A service to start new jobs on a slurm cluster."""

    # Human readable service name
    _service_name = "EMCluster"

    # Logger name
    _logger_name = "cryoemservices.services.cluster"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.schedulers = {}

    def initializing(self):
        """Subscribe to the cluster submission queue.
        Received messages must be acknowledged."""
        self.log.info("Cluster submission service starting")

        self.schedulers = {
            f.name: f.load()
            for f in entry_points(group="cryoemservices.services.cluster.schedulers")
        }
        self.log.info(f"Supported schedulers: {', '.join(self.schedulers.keys())}")
        workflows.recipe.wrap_subscribe(
            self._transport,
            self._environment["queue"] or "cluster.submission",
            self.run_submit_job,
            acknowledgement=True,
            log_extender=self.extend_log,
        )

    def run_submit_job(self, rw, header, message):
        """Submit cluster job according to message."""

        parameters = rw.recipe_step["parameters"]
        cluster_params = JobSubmissionParameters(**parameters.get("cluster", {}))

        if not isinstance(cluster_params.commands, str):
            cluster_params.commands = "\n".join(cluster_params.commands)

        if "recipefile" in parameters:
            recipefile = parameters["recipefile"]
            try:
                Path(recipefile).parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.log.error(f"Cannot make directory for {recipefile}")
                self._transport.nack(header)
                return
            self.log.info("Writing recipe to %s", recipefile)
            cluster_params.commands = cluster_params.commands.replace(
                "$RECIPEFILE", recipefile
            )
            with open(recipefile, "w") as fh:
                fh.write(rw.recipe.pretty())
        if "recipeenvironment" in parameters:
            recipeenvironment = parameters["recipeenvironment"]
            try:
                Path(recipeenvironment).parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.log.error(f"Cannot make directory for {recipeenvironment}")
                self._transport.nack(header)
                return
            self.log.info("Writing recipe environment to %s", recipeenvironment)
            cluster_params.commands = cluster_params.commands.replace(
                "$RECIPEENV", recipeenvironment
            )
            with open(recipeenvironment, "w") as fh:
                json.dump(
                    rw.environment, fh, sort_keys=True, indent=2, separators=(",", ": ")
                )
        if "recipewrapper" in parameters:
            recipewrapper = parameters["recipewrapper"]
            try:
                Path(recipewrapper).parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.log.error(f"Cannot make directory for {recipewrapper}")
                self._transport.nack(header)
                return
            self.log.info("Storing serialized recipe wrapper in %s", recipewrapper)
            cluster_params.commands = cluster_params.commands.replace(
                "$RECIPEWRAP", recipewrapper
            )
            with open(recipewrapper, "w") as fh:
                json.dump(
                    {
                        "recipe": rw.recipe.recipe,
                        "recipe-pointer": rw.recipe_pointer,
                        "environment": rw.environment,
                        "recipe-path": rw.recipe_path,
                        "payload": rw.payload,
                    },
                    fh,
                    indent=2,
                    separators=(",", ": "),
                )

        if "workingdir" not in parameters or not parameters["workingdir"].startswith(
            "/"
        ):
            self.log.error(
                "No absolute working directory specified. Will not run cluster job"
            )
            self._transport.nack(header)
            return
        working_directory = Path(parameters["workingdir"])
        try:
            working_directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.log.error(
                "Could not create working directory: %s", str(e), exc_info=True
            )
            self._transport.nack(header)
            return

        submit_to_scheduler = self.schedulers.get(cluster_params.scheduler)

        service_config = config_from_file(self._environment["config"])
        jobnumber = submit_to_scheduler(
            cluster_params,
            working_directory,
            self.log,
            service_config=service_config,
            cluster_name=self._environment["slurm_cluster"],
        )
        if not jobnumber:
            self.log.error("Job was not submitted")
            self._transport.nack(header)
            return

        # Conditionally acknowledge receipt of the message
        txn = self._transport.transaction_begin(subscription_id=header["subscription"])
        self._transport.ack(header, transaction=txn)

        # Send results onwards
        rw.set_default_channel("job_submitted")
        rw.send({"jobid": jobnumber}, transaction=txn)

        # Commit transaction
        self._transport.transaction_commit(txn)
        self.log.info(f"Submitted job {jobnumber} to slurm")
