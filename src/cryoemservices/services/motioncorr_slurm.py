from __future__ import annotations

import os
from pathlib import Path
from typing import List

from workflows.services.common_service import CommonService

from cryoemservices.services.motioncorr import MotionCorr
from cryoemservices.util.slurm_submission import slurm_submission


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

    def motioncor2(self, command: List[str], mrc_out: Path):
        """Submit MotionCor2 jobs to the slurm cluster via the RestAPI"""
        slurm_outcome = slurm_submission(
            log=self.log,
            job_name="MotionCorr",
            command=command,
            project_dir=mrc_out.parent,
            output_file=mrc_out,
            cpus=1,
            use_gpu=True,
            use_singularity=True,
            cif_name=os.environ["MOTIONCOR2_SIF"],
        )

        if not slurm_outcome.returncode:
            # Read in the output logs
            slurm_output_file = f"{mrc_out}.out"
            slurm_error_file = f"{mrc_out}.err"
            submission_file = f"{mrc_out}.json"
            if Path(slurm_output_file).is_file():
                self.parse_mc_slurm_output(slurm_output_file)

            # Clean up if everything succeeded
            if self.x_shift_list and self.y_shift_list:
                Path(slurm_output_file).unlink()
                Path(slurm_error_file).unlink()
                Path(submission_file).unlink()
            else:
                self.log.error(f"Reading shifts from {slurm_output_file} failed")
                slurm_outcome.returncode = 1
        return slurm_outcome
