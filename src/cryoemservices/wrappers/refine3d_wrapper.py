from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import zocalo.wrapper
from pydantic import BaseModel, Field, ValidationError

from cryoemservices.util.spa_relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)

logger = logging.getLogger("relion.refine.wrapper")


class RefineParameters(BaseModel):
    refine_job_dir: str = Field(..., min_length=1)
    particles_file: str = Field(..., min_length=1)
    rescaled_class_reference: str = Field(..., min_length=1)
    is_first_refinement: bool
    number_of_particles: int
    batch_size: int
    pixel_size: float
    class_number: int
    mask_diameter: float
    mask: Optional[str] = None
    mask_lowpass: float = 15
    mask_threshold: float = 0.02
    mask_extend: int = 3
    mask_soft_edge: int = 3
    mpi_run_command: str = "srun -n 5"
    dont_correct_greyscale: bool = True
    ini_high: float = 20.0
    dont_combine_weights_via_disc: bool = True
    nr_pool: int = 10
    pad: int = 2
    do_ctf: bool = True
    ctf_intact_first_peak: bool = False
    flatten_solvent: bool = True
    do_zero_mask: bool = True
    oversampling: int = 1
    healpix_order: int = 2
    local_healpix_order: int = 4
    low_resol_join_halves: int = 40
    offset_range: float = 5
    offset_step: float = 2
    ignore_angles: bool = False
    resol_angles: bool = False
    symmetry: str = "C1"
    do_norm: bool = True
    do_scale: bool = True
    threads: int = 8
    gpus: str = "0:1:2:3"
    relion_options: RelionServiceOptions


class RefineWrapper(zocalo.wrapper.BaseWrapper):
    """
    A wrapper for the Relion 3D refinement pipeline.
    """

    # Job names
    refine_job_type = "relion.refine3d"
    mask_job_type = "relion.maskcreate"

    def run(self):
        """
        Run 3D refinement and postprocessing
        """
        assert hasattr(self, "recwrap"), "No recipewrapper object found"
        params_dict = self.recwrap.recipe_step["job_parameters"]
        try:
            refine_params = RefineParameters(**params_dict)
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Refinement parameter validation failed for parameters: {params_dict} "
                f"with exception: {e}"
            )
            return False

        # Determine the directory to run in and the job numbers
        self.log.info(f"Running refinement pipeline for {refine_params.particles_file}")
        project_dir = Path(refine_params.refine_job_dir).parent.parent
        os.chdir(project_dir)

        job_num_refine = int(
            re.search("/job[0-9]{3}", refine_params.refine_job_dir)[0][4:7]
        )
        job_num_postprocess = (
            job_num_refine + 1 if refine_params.mask else job_num_refine + 2
        )

        # Update the relion options
        refine_params.relion_options = update_relion_options(
            refine_params.relion_options, dict(refine_params)
        )

        # Set up the command for the refinement job
        Path(refine_params.refine_job_dir).mkdir(parents=True, exist_ok=True)
        self.log.info(
            f"Running {self.refine_job_type} in {refine_params.refine_job_dir}"
        )
        refine_command = refine_params.mpi_run_command.split()
        refine_command.extend(
            [
                "relion_refine_mpi",
                "--i",
                str(refine_params.particles_file),
                "--o",
                f"{refine_params.refine_job_dir}/run",
                "--ref",
                str(refine_params.rescaled_class_reference),
                "--particle_diameter",
                f"{refine_params.mask_diameter}",
                "--auto_refine",
                "--split_random_halves",
            ]
        )

        # Add flags to the command based on the input parameters
        refine_flags = {
            "dont_correct_greyscale": "--firstiter_cc",
            "ini_high": "--ini_high",
            "dont_combine_weights_via_disc": "--dont_combine_weights_via_disc",
            "nr_pool": "--pool",
            "pad": "--pad",
            "do_ctf": "--ctf",
            "ctf_intact_first_peak": "--ctf_intact_first_peak",
            "flatten_solvent": "--flatten_solvent",
            "do_zero_mask": "--zero_mask",
            "oversampling": "--oversampling",
            "healpix_order": "--healpix_order",
            "local_healpix_order": "--auto_local_healpix_order",
            "low_resol_join_halves": "--low_resol_join_halves",
            "offset_range": "--offset_range",
            "offset_step": "--offset_step",
            "ignore_angles": "--auto_ignore_angles",
            "resol_angles": "--auto_resol_angles",
            "symmetry": "--sym",
            "do_norm": "--norm",
            "do_scale": "--scale",
            "threads": "--j",
            "gpus": "--gpu",
        }
        for k, v in refine_params.dict().items():
            if v and (k in refine_flags):
                if type(v) is bool:
                    refine_command.append(refine_flags[k])
                else:
                    refine_command.extend((refine_flags[k], str(v)))
        refine_command.extend(
            ("--pipeline_control", f"{refine_params.refine_job_dir}/")
        )

        # Run Refine3D and confirm it ran successfully
        refine_result = subprocess.run(refine_command, capture_output=True)

        # Register the Refine3D job with the node creator
        self.log.info(f"Sending {self.refine_job_type} to node creator")
        node_creator_parameters = {
            "job_type": self.refine_job_type,
            "input_file": f"{refine_params.particles_file}:{refine_params.rescaled_class_reference}",
            "output_file": f"{refine_params.refine_job_dir}/",
            "relion_options": dict(refine_params.relion_options),
            "command": " ".join(refine_command),
            "stdout": refine_result.stdout.decode("utf8", "replace"),
            "stderr": refine_result.stderr.decode("utf8", "replace"),
        }
        if refine_result.returncode:
            node_creator_parameters["success"] = False
        else:
            node_creator_parameters["success"] = True
        self.recwrap.send_to("node_creator", node_creator_parameters)

        # End here if the command failed
        if refine_result.returncode:
            self.log.error(
                "Refinement Refine3D failed with exitcode "
                f"{refine_result.returncode}:\n"
                + refine_result.stderr.decode("utf8", "replace")
            )
            return False

        # Do the mask creation if one isn't provided
        mask_job_dir = Path(f"MaskCreate/job{job_num_refine + 1:03}")
        if not refine_params.mask:
            self.log.info(f"Running {self.mask_job_type} in {mask_job_dir}")
            mask_job_dir.mkdir(parents=True, exist_ok=True)
            mask_command = [
                "relion_mask_create",
                "--i",
                f"{refine_params.refine_job_dir}/run_class001.mrc",
                "--o",
                f"{mask_job_dir}/mask.mrc",
                "--lowpass",
                str(refine_params.mask_lowpass),
                "--ini_threshold",
                str(refine_params.mask_threshold),
                "--extend_inimask",
                str(refine_params.mask_extend),
                "--width_soft_edge",
                str(refine_params.mask_soft_edge),
                "--angpix",
                str(refine_params.pixel_size),
                "--j",
                str(refine_params.threads),
                "--pipeline_control",
                f"{mask_job_dir}/",
            ]
            mask_result = subprocess.run(mask_command, capture_output=True)

            # Register the mask creation job with the node creator
            self.log.info(f"Sending {self.mask_job_type} to node creator")

            node_creator_mask = {
                "job_type": self.mask_job_type,
                "input_file": f"{refine_params.refine_job_dir}/run_class001.mrc",
                "output_file": f"{project_dir}/{mask_job_dir}/mask.mrc",
                "relion_options": dict(refine_params.relion_options),
                "command": " ".join(mask_command),
                "stdout": mask_result.stdout.decode("utf8", "replace"),
                "stderr": mask_result.stderr.decode("utf8", "replace"),
            }
            if mask_result.returncode:
                node_creator_mask["success"] = False
            else:
                node_creator_mask["success"] = True
            self.recwrap.send_to("node_creator", node_creator_mask)

            # End here if the command failed
            if mask_result.returncode:
                self.log.error(
                    "Refinement mask creation failed with exitcode "
                    f"{mask_result.returncode}:\n"
                    + mask_result.stderr.decode("utf8", "replace")
                )
                return False

        # Send to postprocessing
        postprocess_params = {
            "half_map": f"{refine_params.refine_job_dir}/run_half1_class001_unfil.mrc",
            "mask": refine_params.mask
            if refine_params.mask
            else f"{mask_job_dir}/mask.mrc",
            "rescaled_class_reference": refine_params.rescaled_class_reference,
            "job_dir": f"PostProcess/job{job_num_postprocess:03}",
            "is_first_refinement": refine_params.is_first_refinement,
            "pixel_size": refine_params.pixel_size,
            "number_of_particles": refine_params.number_of_particles,
            "batch_size": refine_params.batch_size,
            "class_number": refine_params.class_number,
            "symmetry": refine_params.symmetry,
            "relion_options": refine_params.relion_options,
        }
        self.recwrap.send_to("postprocess", postprocess_params)
        return True
