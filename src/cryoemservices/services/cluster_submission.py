from __future__ import annotations

import datetime
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import requests
import workflows.recipe
import zocalo.configuration
from importlib_metadata import entry_points
from pydantic import BaseModel, Field
from workflows.services.common_service import CommonService
from zocalo.util import slurm


class JobSubmissionParameters(BaseModel):
    scheduler: str = "slurm"
    partition: Optional[str]
    prefer: Optional[str]
    job_name: Optional[str]
    environment: Optional[dict[str, str]] = None
    cpus_per_task: Optional[int] = None
    tasks: Optional[int] = None
    nodes: Optional[int]
    memory_per_node: Optional[int] = None
    gpus_per_node: Optional[str] = None
    min_memory_per_cpu: Optional[int] = Field(
        None, description="Minimum real memory per cpu (MB)"
    )
    time_limit: Optional[datetime.timedelta] = None
    gpus: Optional[int] = None
    exclusive: bool = False
    commands: str | list[str]
    qos: Optional[str]


def submit_to_slurm(
    params: JobSubmissionParameters,
    working_directory: Path,
    logger: logging.Logger,
    zc: zocalo.configuration,
) -> int | None:
    api = slurm.SlurmRestApi.from_zocalo_configuration(zc)

    script = params.commands
    if not isinstance(script, str):
        script = "\n".join(script)
    script = f"#!/bin/bash\n. /etc/profile.d/modules.sh\n{script}"

    logger.debug(f"Submitting script to Slurm:\n{script}")
    if params.time_limit:
        time_limit_minutes = math.ceil(params.time_limit.total_seconds() / 60)
    else:
        time_limit_minutes = None
    job_submission = slurm.models.JobSubmission(
        script=script,
        job=slurm.models.JobProperties(
            partition=params.partition,
            prefer=params.prefer,
            name=params.job_name,
            cpus_per_task=params.cpus_per_task,
            tasks=params.tasks,
            nodes=[params.nodes, params.nodes] if params.nodes else params.nodes,
            gpus_per_node=params.gpus_per_node,
            memory_per_node=params.memory_per_node,
            environment=os.environ
            if params.environment is None
            else params.environment,
            memory_per_cpu=params.min_memory_per_cpu,
            time_limit=time_limit_minutes,
            gpus=params.gpus,
            current_working_directory=os.fspath(working_directory),
            qos=params.qos,
        ),
    )
    try:
        response = api.submit_job(job_submission)
    except requests.HTTPError as e:
        logger.error(f"Failed Slurm job submission: {e}\n" f"{e.response.text}")
        return None
    if response.errors:
        error_message = "\n".join(f"{e.errno}: {e.error}" for e in response.errors)
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
        self.log.debug(f"Supported schedulers: {', '.join(self.schedulers.keys())}")
        workflows.recipe.wrap_subscribe(
            self._transport,
            "cluster.submission",
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
            self.log.debug("Writing recipe to %s", recipefile)
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
            self.log.debug("Writing recipe environment to %s", recipeenvironment)
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
            self.log.debug("Storing serialized recipe wrapper in %s", recipewrapper)
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

        jobnumber = submit_to_scheduler(
            cluster_params, working_directory, self.log, zc=self.config
        )
        if not jobnumber:
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
