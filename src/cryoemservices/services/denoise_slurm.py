from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List

from cryoemservices.services.denoise import Denoise
from cryoemservices.services.tomo_align_slurm import retrieve_files, transfer_files
from cryoemservices.util.slurm_submission import slurm_submission


class DenoiseSlurm(Denoise):
    """
    A service for denoising cryoEM tomograms using Topaz
    Submits jobs to a slurm cluster via RestAPI
    """

    # Logger name
    _logger_name = "cryoemservices.services.denoise_slurm"

    def run_topaz(
        self,
        topaz_command: List[str],
        alignment_output_dir: Path,
        tomogram_volume: Path,
        denoised_full_path: Path,
    ):
        """Submit topaz jobs to a slurm cluster"""

        # Transfer the required files
        self.log.info("Transferring files...")
        items_to_transfer = [tomogram_volume]
        transfer_status = transfer_files(items_to_transfer)
        if len(transfer_status) != len(items_to_transfer):
            self.log.error(
                "Unable to transfer files: "
                f"desired {items_to_transfer}, done {transfer_status}"
            )
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout="".encode("utf8"),
                stderr="Failed transfer".encode("utf8"),
            )
        self.log.info("All files transferred")

        # Submit the command to slurm
        self.log.info(f"Running topaz denoising with command: {topaz_command}")
        slurm_outcome = slurm_submission(
            log=self.log,
            service_config_file=self._environment["config"],
            slurm_cluster=self._environment["slurm_cluster"],
            job_name="Denoising",
            command=topaz_command,
            project_dir=alignment_output_dir,
            output_file=denoised_full_path,
            cpus=1,
            use_gpu=True,
            use_singularity=True,
            cif_name=os.environ["DENOISING_SIF"],
            external_filesystem=True,
        )

        # Get back any output files and clean up
        self.log.info("Retrieving output files...")
        retrieve_files(
            job_directory=alignment_output_dir,
            files_to_skip=[tomogram_volume],
            basepath=str(tomogram_volume.stem),
        )
        self.log.info("All output files retrieved")

        slurm_output_file = f"{denoised_full_path}.out"
        slurm_error_file = f"{denoised_full_path}.out"

        if not denoised_full_path.is_file():
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout="".encode("utf8"),
                stderr=f"Output {denoised_full_path} not found".encode("utf8"),
            )

        try:
            with open(slurm_output_file, "r") as slurm_stdout:
                slurm_outcome.stdout = slurm_stdout.read()
            with open(slurm_error_file, "r") as slurm_stderr:
                slurm_outcome.stderr = slurm_stderr.read()
        except FileNotFoundError:
            self.log.error(f"Output file {slurm_output_file} not found")
            slurm_outcome.stdout = ""
            slurm_outcome.stderr = f"Reading output file {slurm_output_file} failed"
        slurm_outcome.stdout = slurm_outcome.stdout.encode("utf8")
        slurm_outcome.stderr = slurm_outcome.stderr.encode("utf8")
        return slurm_outcome
