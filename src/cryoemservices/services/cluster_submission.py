from __future__ import annotations

import json
from pathlib import Path

from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.config import config_from_file
from cryoemservices.util.slurm_submission import (
    JobSubmissionParameters,
    submit_to_slurm,
)


class ClusterSubmission(CommonService):
    """A service to start new jobs on a slurm cluster."""

    # Logger name
    _logger_name = "cryoemservices.services.cluster"

    def initializing(self):
        """Subscribe to the cluster submission queue.
        Received messages must be acknowledged."""
        self.log.info("Cluster submission service starting for slurm")
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "cluster.submission",
            self.run_submit_job,
            acknowledgement=True,
        )

    def run_submit_job(self, rw, header, message):
        """Submit cluster job according to message."""

        parameters = rw.recipe_step["parameters"]
        if type(parameters.get("cluster", {}).get("commands")) is list:
            parameters["cluster"]["commands"] = "\n".join(
                parameters["cluster"]["commands"]
            )
        cluster_params = JobSubmissionParameters(**parameters.get("cluster", {}))

        if "wrapper" in parameters:
            wrapper = parameters["wrapper"]
            try:
                Path(wrapper).parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.log.error(f"Cannot make directory for {wrapper}")
                self._transport.nack(header)
                return
            self.log.info(f"Storing serialized recipe wrapper in {wrapper}")
            cluster_params.commands = cluster_params.commands.replace(
                "$RECIPEWRAP", wrapper
            )
            with open(wrapper, "w") as fh:
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

        if parameters.get("standard_output"):
            stdout_file = Path(parameters["standard_output"])
        else:
            stdout_file = working_directory / "run.out"
        if parameters.get("standard_error"):
            stderr_file = Path(parameters["standard_error"])
        else:
            stderr_file = working_directory / "run.err"

        service_config = config_from_file(self._environment["config"])
        jobnumber = submit_to_slurm(
            params=cluster_params,
            working_directory=working_directory,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            logger=self.log,
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
