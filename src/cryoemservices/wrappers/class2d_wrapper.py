from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from gemmi import cif
from pydantic import BaseModel, Field, ValidationError

from cryoemservices.util.relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)

logger = logging.getLogger("cryoemservices.wrappers.class2d")


class Class2DParameters(BaseModel):
    particles_file: str = Field(..., min_length=1)
    class2d_dir: str = Field(..., min_length=1)
    batch_is_complete: bool
    batch_size: int
    particle_diameter: float = 0
    mask_diameter: float = 190
    do_vdam: bool = True
    vdam_write_iter: int = 10
    vdam_threshold: float = 0.1
    vdam_mini_batches: int = 150
    vdam_subset: int = 7000
    vdam_initial_fraction: float = 0.3
    vdam_final_fraction: float = 0.1
    dont_combine_weights_via_disc: bool = True
    preread_images: bool = True
    scratch_dir: Optional[str] = None
    nr_pool: int = 100
    pad: int = 2
    skip_gridding: bool = False
    do_ctf: bool = True
    ctf_intact_first_peak: bool = False
    class2d_nr_iter: int = 20
    tau_fudge: float = 2
    class2d_nr_classes: int = 50
    flatten_solvent: bool = True
    do_zero_mask: bool = True
    highres_limit: Optional[float] = None
    centre_classes: bool = True
    oversampling: int = 1
    skip_align: bool = False
    psi_step: float = 12.0
    offset_range: float = 5
    offset_step: float = 2
    allow_coarser: bool = False
    do_norm: bool = True
    do_scale: bool = True
    mpi_run_command: str = "srun -n 5"
    threads: int = 8
    gpus: str = "0"
    picker_id: int
    class2d_grp_uuid: int
    class_uuids: str
    do_icebreaker_jobs: bool = True
    do_cryodann: bool = False
    cryodann_dataset: str = ""
    relion_options: RelionServiceOptions


def run_cryodann(
    class2d_params: Class2DParameters, project_dir: Path, nr_iter: int
) -> bool:
    # cryoVAE needs the particles to be aligned according to the alignments determined in 2D classification
    (Path(class2d_params.class2d_dir) / "aligned_particles").mkdir()
    particle_alignment_command = [
        "relion_stack_create",
        "--i",
        f"{class2d_params.class2d_dir}/run_it{nr_iter:03}_data.star",
        "--o",
        f"{class2d_params.class2d_dir}/aligned_particles/aligned",
    ]
    aligned_result = subprocess.run(
        particle_alignment_command, cwd=str(project_dir), capture_output=True
    )
    if aligned_result.returncode:
        logger.error(
            f"Failed to align particles for {class2d_params.class2d_dir}, class selection alone will be used for filtering"
        )
        return False

    # now need to low-pass the particle images
    lowpass_command = [
        "relion_image_handler",
        "--i",
        f"{class2d_params.class2d_dir}/aligned_particles/aligned.mrcs",
        "--lowpass",
        "10",
        "--o",
        f"{class2d_params.class2d_dir}/aligned_particles/aligned_lowpassed.mrcs",
    ]
    lowpass_result = subprocess.run(
        lowpass_command, cwd=str(project_dir), capture_output=True
    )
    if lowpass_result.returncode:
        logger.error(
            f"Failed to lowpass particles for {class2d_params.class2d_dir}, class selection alone will be used for filtering"
        )
        return False

    # now the aligned and lowpassed particles can be fed into cryoVAE
    (Path(class2d_params.class2d_dir) / "cryodann").mkdir(exist_ok=True)
    cryovae_command = [
        "cryovae",
        f"{class2d_params.class2d_dir}/aligned_particles/aligned_lowpassed.mrcs",
        f"{class2d_params.class2d_dir}/cryodann/cryovae",
        "--beta=0.1",
    ]
    cryovae_result = subprocess.run(
        cryovae_command, cwd=str(project_dir), capture_output=True
    )
    if cryovae_result.returncode:
        logger.error(
            f"Failed to run cryoVAE for {class2d_params.class2d_dir}, class selection alone will be used for filtering"
        )
        return False
    # then run cryodann on the denoised output
    cryodann_command = [
        "cryodann",
        class2d_params.cryodann_dataset,
        f"{class2d_params.class2d_dir}/cryodann/cryovae/recons.mrcs",
        f"{class2d_params.class2d_dir}/cryodann",
        "--particle_file",
        f"{class2d_params.class2d_dir}/aligned_particles/aligned.star",
        "--keep_percent",
        "0.5",
    ]
    cryodann_result = subprocess.run(
        cryodann_command, cwd=str(project_dir), capture_output=True
    )
    if cryodann_result.returncode:
        logger.error(
            f"Failed to run cryoDANN for {class2d_params.class2d_dir}, class selection alone will be used for filtering"
        )
        return False
    return True


def run_class2d(class2d_params: Class2DParameters, send_to_rabbitmq: Callable):
    # Class ids get fed in as a string, need to convert these to a dictionary
    class_uuids_dict = json.loads(class2d_params.class_uuids.replace("'", '"'))
    class_uuids_keys = list(class_uuids_dict.keys())

    if class2d_params.do_vdam:
        job_type = "relion.class2d.vdam"
    else:
        job_type = "relion.class2d.em"

    # Update the relion options to get out the mask diameter
    class2d_params.relion_options = update_relion_options(
        class2d_params.relion_options, dict(class2d_params)
    )

    # Make the job directory and move to the project directory
    job_dir = Path(class2d_params.class2d_dir)
    if (job_dir / "RELION_JOB_EXIT_SUCCESS").exists():
        # This job over-writes a previous one
        job_is_rerun = True
    else:
        job_is_rerun = False
        job_dir.mkdir(parents=True, exist_ok=True)
    project_dir = job_dir.parent.parent
    os.chdir(project_dir)

    job_num_search = re.search("/job[0-9]+", class2d_params.class2d_dir)
    if job_num_search:
        job_num = int(job_num_search[0][4:7])
    else:
        logger.warning(f"Invalid job directory in {class2d_params.class2d_dir}")
        return False

    particles_file = str(Path(class2d_params.particles_file).relative_to(project_dir))
    logger.info(f"Running Class2D for {particles_file}")

    class2d_flags = {
        "dont_combine_weights_via_disc": "--dont_combine_weights_via_disc",
        "preread_images": "--preread_images",
        "scratch_dir": "--scratch_dir",
        "nr_pool": "--pool",
        "pad": "--pad",
        "skip_gridding": "--skip_gridding",
        "do_ctf": "--ctf",
        "ctf_intact_first_peak": "--ctf_intact_first_peak",
        "tau_fudge": "--tau2_fudge",
        "class2d_nr_classes": "--K",
        "flatten_solvent": "--flatten_solvent",
        "do_zero_mask": "--zero_mask",
        "highres_limit": "--strict_highres_exp",
        "centre_classes": "--center_classes",
        "oversampling": "--oversampling",
        "skip_align": "--skip_align",
        "psi_step": "--psi_step",
        "offset_range": "--offset_range",
        "offset_step": "--offset_step",
        "allow_coarser": "--allow_coarser_sampling",
        "do_norm": "--norm",
        "do_scale": "--scale",
        "threads": "--j",
        "gpus": "--gpu",
    }

    # Create the classification command
    if class2d_params.do_vdam:
        class2d_command = ["relion_refine"]
    else:
        class2d_command = class2d_params.mpi_run_command.split()
        class2d_command.append("relion_refine_mpi")
    class2d_command.extend(
        [
            "--i",
            particles_file,
            "--o",
            f"{job_dir.relative_to(project_dir)}/run",
            "--particle_diameter",
            f"{class2d_params.relion_options.mask_diameter}",
        ]
    )
    for k, v in class2d_params.model_dump().items():
        if k in class2d_flags:
            if (type(v) is not bool) and (v not in [None, ""]):
                class2d_command.extend((class2d_flags[k], str(v)))
            elif v:
                class2d_command.append(class2d_flags[k])
    class2d_command.extend(
        ("--pipeline_control", f"{job_dir.relative_to(project_dir)}/")
    )
    if class2d_params.do_vdam:
        class2d_command.extend(
            (
                "--grad",
                "--class_inactivity_threshold",
                str(class2d_params.vdam_threshold),
                "--grad_write_iter",
                str(class2d_params.vdam_write_iter),
                "--grad_fin_subset",
                str(class2d_params.vdam_subset),
                "--grad_ini_frac",
                str(class2d_params.vdam_initial_fraction),
                "--grad_fin_frac",
                str(class2d_params.vdam_final_fraction),
            )
        )
        nr_iter = class2d_params.vdam_mini_batches
    else:
        nr_iter = class2d_params.class2d_nr_iter
    class2d_command.extend(("--iter", str(nr_iter)))

    # Run Class2D and confirm it ran successfully
    logger.info(" ".join(class2d_command))
    result = subprocess.run(class2d_command, cwd=str(project_dir), capture_output=True)
    if not job_is_rerun:
        (job_dir / "RELION_JOB_EXIT_SUCCESS").unlink(missing_ok=True)

    # Register the Class2D job with the node creator
    logger.info(f"Sending {job_type} to node creator")
    node_creator_parameters: dict = {
        "job_type": job_type,
        "input_file": class2d_params.particles_file,
        "output_file": class2d_params.class2d_dir,
        "relion_options": dict(class2d_params.relion_options),
        "command": " ".join(class2d_command),
        "stdout": result.stdout.decode("utf8", "replace"),
        "stderr": result.stderr.decode("utf8", "replace"),
        "success": (result.returncode == 0),
    }
    send_to_rabbitmq("node_creator", node_creator_parameters)

    # End here if the command failed
    if result.returncode:
        logger.error(
            f"Relion Class2D failed with exitcode {result.returncode}:\n"
            + result.stderr.decode("utf8", "replace")
        )
        # Tell Murfey to release the hold on 2D classification jobs
        murfey_params = {
            "register": "done_incomplete_2d_batch",
            "job_dir": class2d_params.class2d_dir,
        }
        send_to_rabbitmq("murfey_feedback", murfey_params)
        return False

    # Send classification job information to ispyb
    class_particles_file = cif.read_file(
        f"{class2d_params.class2d_dir}/run_it{nr_iter:03}_data.star"
    )
    optics_block = class_particles_file.find_block("optics")
    binned_pixel_size = optics_block.find_loop("_rlnImagePixelSize")[0]
    if not class2d_params.batch_is_complete:
        particles_block = class_particles_file.find_block("particles")
        particles_in_batch = len(particles_block.find_loop("_rlnCoordinateX"))
    else:
        particles_in_batch = class2d_params.batch_size

    ispyb_parameters = []
    classification_grp_ispyb_parameters = {
        "ispyb_command": "buffer",
        "buffer_command": {"ispyb_command": "insert_particle_classification_group"},
        "type": "2D",
        "batch_number": int(
            class2d_params.particles_file.split("particles_split")[1].split(".")[0]
        ),
        "number_of_particles_per_batch": particles_in_batch,
        "number_of_classes_per_batch": class2d_params.class2d_nr_classes,
        "symmetry": "C1",
        "binned_pixel_size": binned_pixel_size,
        "particle_picker_id": class2d_params.picker_id,
    }
    if job_is_rerun:
        # If this job overwrites another get the id for it
        classification_grp_ispyb_parameters["buffer_lookup"] = {
            "particle_classification_group_id": class2d_params.class2d_grp_uuid,
        }
    else:
        classification_grp_ispyb_parameters["buffer_store"] = (
            class2d_params.class2d_grp_uuid
        )
    ispyb_parameters.append(classification_grp_ispyb_parameters)

    # Send individual classes to ispyb
    class_star_file = cif.read_file(
        f"{class2d_params.class2d_dir}/run_it{nr_iter:03}_model.star"
    )
    classes_block = class_star_file.find_block("model_classes")
    classes_loop = classes_block.find_loop("_rlnReferenceImage").get_loop()
    if class2d_params.do_vdam:
        vdam_offset = 2
    else:
        vdam_offset = 0

    for class_id in range(class2d_params.class2d_nr_classes):
        # Add an ispyb insert for each class
        class_ispyb_parameters = {
            "ispyb_command": "buffer",
            "buffer_lookup": {
                "particle_classification_group_id": class2d_params.class2d_grp_uuid
            },
            "buffer_command": {"ispyb_command": "insert_particle_classification"},
            "class_number": class_id + 1,
            "class_image_full_path": (
                f"{class2d_params.class2d_dir}"
                f"/run_it{nr_iter:03}_classes_{class_id+1}.jpeg"
            ),
            "particles_per_class": (
                float(classes_loop[class_id, 1 + vdam_offset]) * particles_in_batch
            ),
            "class_distribution": classes_loop[class_id, 1 + vdam_offset],
            "rotation_accuracy": classes_loop[class_id, 2 + vdam_offset],
            "translation_accuracy": classes_loop[class_id, 3 + vdam_offset],
        }
        if job_is_rerun:
            class_ispyb_parameters["buffer_lookup"].update(
                {
                    "particle_classification_id": class_uuids_dict[
                        class_uuids_keys[class_id]
                    ]
                }
            )
        else:
            class_ispyb_parameters["buffer_store"] = class_uuids_dict[
                class_uuids_keys[class_id]
            ]

        # Add the resolution and fourier completeness if they are valid numbers
        estimated_resolution = float(classes_loop[class_id, 4 + vdam_offset])
        if np.isfinite(estimated_resolution):
            class_ispyb_parameters["estimated_resolution"] = estimated_resolution
        else:
            class_ispyb_parameters["estimated_resolution"] = 0.0
        fourier_completeness = float(classes_loop[class_id, 5 + vdam_offset])
        if np.isfinite(fourier_completeness):
            class_ispyb_parameters["overall_fourier_completeness"] = (
                fourier_completeness
            )
        else:
            class_ispyb_parameters["overall_fourier_completeness"] = 0.0

        # Add the ispyb command to the command list
        ispyb_parameters.append(class_ispyb_parameters)

    # Send a request to make the class images
    logger.info("Sending to images service")
    send_to_rabbitmq(
        "images",
        {
            "image_command": "mrc_to_jpeg",
            "file": (f"{class2d_params.class2d_dir}/run_it{nr_iter:03}_classes.mrcs"),
            "all_frames": "True",
        },
    )

    # Send all the ispyb class insertion commands
    logger.info(f"Sending to ispyb {ispyb_parameters}")
    send_to_rabbitmq(
        "ispyb_connector",
        {
            "ispyb_command": "multipart_message",
            "ispyb_command_list": ispyb_parameters,
        },
    )

    if class2d_params.batch_is_complete:
        # Create an icebreaker job
        if class2d_params.do_icebreaker_jobs:
            logger.info("Sending to icebreaker particle analysis")
            icebreaker_params = {
                "icebreaker_type": "particles",
                "input_micrographs": (
                    f"{project_dir}/IceBreaker/job003/grouped_micrographs.star"
                ),
                "input_particles": class2d_params.particles_file,
                "output_path": f"{project_dir}/IceBreaker/job{job_num + 1:03}/",
                "mc_uuid": -1,
                "relion_options": dict(class2d_params.relion_options),
            }
            send_to_rabbitmq("icebreaker", icebreaker_params)

        if class2d_params.do_cryodann:
            cryodann_success = run_cryodann(class2d_params, project_dir, nr_iter)
            if cryodann_success:
                lightning_log_dir = (
                    project_dir
                    / class2d_params.class2d_dir
                    / "cryodann"
                    / "lightning_logs"
                )
                if (lightning_log_dir / "scores.npy").is_file():
                    lightning_log_scores = lightning_log_dir / "scores.npy"
                else:
                    lightning_log_options = lightning_log_dir.glob("*/scores.npy")
                    lightning_log_scores = None
                    for lightning_file in lightning_log_options:
                        lightning_log_scores = lightning_file

                if lightning_log_scores:
                    cryodann_scores = np.load(lightning_log_scores).flatten()
                    cryodann_block = class_particles_file.find_block("particles")
                    cryodann_loop = cryodann_block.find_loop(
                        "_rlnCoordinateX"
                    ).get_loop()
                    cryodann_loop.add_columns(["_rlnCryodannScore"], "0")
                    for i in range(cryodann_loop.length()):
                        cryodann_loop[i, -1] = str(cryodann_scores[i])
                    class_particles_file.write_file(
                        f"{class2d_params.class2d_dir}/run_it{nr_iter:03}_data.star"
                    )
                else:
                    logger.error("Cryodann ran but no scores have been found")

        # Create a 2D autoselection job
        logger.info("Sending to class selection")
        autoselect_parameters = {
            "input_file": f"{class2d_params.class2d_dir}/run_it{nr_iter:03}_optimiser.star",
            "relion_options": dict(class2d_params.relion_options),
            "class_uuids": class2d_params.class_uuids,
        }
        send_to_rabbitmq("select_classes", autoselect_parameters)
    else:
        # Tell Murfey the incomplete batch has finished
        murfey_params = {
            "register": "done_incomplete_2d_batch",
            "job_dir": class2d_params.class2d_dir,
        }
        send_to_rabbitmq("murfey_feedback", murfey_params)

    (job_dir / "RELION_JOB_EXIT_SUCCESS").touch(exist_ok=True)
    logger.info(f"Done {job_type} for {class2d_params.particles_file}.")
    return True


class Class2DWrapper:
    """
    A wrapper for the Relion 2D classification job.
    """

    # Values to extract for ISPyB
    previous_total_count = 0
    total_count = 0

    def __init__(self, recwrap):
        self.log = logging.LoggerAdapter(logger)
        self.recwrap = recwrap

    def run(self):
        """
        Run the 2D classification and register results
        """
        params_dict = self.recwrap.recipe_step["job_parameters"]
        try:
            class2d_params = Class2DParameters(**params_dict)
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Class2D parameter validation failed for parameters: {params_dict} "
                f"with exception: {e}"
            )
            return False

        successful_run = run_class2d(
            class2d_params, send_to_rabbitmq=self.recwrap.send_to
        )
        if not successful_run:
            return False
        return True
