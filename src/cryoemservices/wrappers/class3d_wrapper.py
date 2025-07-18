from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from gemmi import cif
from pydantic import BaseModel, Field, ValidationError

from cryoemservices.pipeliner_plugins.angular_distribution_plot import (
    angular_distribution_plot,
)
from cryoemservices.pipeliner_plugins.angular_efficiency import find_efficiency
from cryoemservices.util.relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)

logger = logging.getLogger("cryoemservices.wrappers.class3d")

job_type = "relion.class3d"

class3d_common_flags = {
    "dont_combine_weights_via_disc": "--dont_combine_weights_via_disc",
    "preread_images": "--preread_images",
    "scratch_dir": "--scratch_dir",
    "nr_pool": "--pool",
    "pad": "--pad",
    "skip_gridding": "--skip_gridding",
    "do_ctf": "--ctf",
    "ctf_intact_first_peak": "--ctf_intact_first_peak",
    "class3d_nr_classes": "--K",
    "flatten_solvent": "--flatten_solvent",
    "do_zero_mask": "--zero_mask",
    "oversampling": "--oversampling",
    "healpix_order": "--healpix_order",
    "threads": "--j",
}


class Class3DParameters(BaseModel):
    particles_file: str = Field(..., min_length=1)
    class3d_dir: str = Field(..., min_length=1)
    batch_size: int
    particle_diameter: float = 0
    mask_diameter: float = 190
    do_initial_model: bool = False
    initial_model_file: Optional[str] = None
    initial_model_iterations: int = 200
    initial_model_offset_range: float = 6
    initial_model_offset_step: float = 2
    start_initial_model_C1: bool = True
    dont_combine_weights_via_disc: bool = True
    preread_images: bool = True
    scratch_dir: Optional[str] = None
    nr_pool: int = 10
    pad: int = 2
    skip_gridding: bool = False
    dont_correct_greyscale: bool = True
    initial_lowpass: float = 40.0
    do_ctf: bool = True
    ctf_intact_first_peak: bool = False
    class3d_nr_iter: int = 20
    fast_subsets: bool = False
    tau_fudge: float = 4
    class3d_nr_classes: int = 4
    flatten_solvent: bool = True
    do_zero_mask: bool = True
    highres_limit: Optional[float] = None
    fn_mask: Optional[str] = None
    oversampling: int = 1
    skip_align: bool = False
    healpix_order: int = 2
    offset_range: float = 5
    offset_step: float = 4
    allow_coarser: bool = False
    symmetry: str = "C1"
    do_norm: bool = True
    do_scale: bool = True
    mpi_run_command: str = "srun -n 5"
    threads: int = 8
    gpus: str = "0:1:2:3"
    initial_model_gpus: str = "0,1,2,3"
    picker_id: int
    class3d_grp_uuid: int
    class_uuids: str
    relion_options: RelionServiceOptions


def run_initial_model(
    initial_model_params: Class3DParameters,
    project_dir: Path,
    job_num: int,
    send_to_rabbitmq: Callable,
) -> Tuple[str, list[dict]]:
    """
    Run the initial model for 3D classification and register results
    """
    # Class ids get fed in as a string, need to convert these to a dictionary
    class_uuids_dict = json.loads(initial_model_params.class_uuids.replace("'", '"'))
    class_uuids_keys = list(class_uuids_dict.keys())

    # Make the job directory and move to the project directory
    job_dir = project_dir / f"InitialModel/job{job_num:03}/"
    job_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(project_dir)
    particles_file = str(
        Path(initial_model_params.particles_file).relative_to(project_dir)
    )

    initial_model_flags = {
        "initial_model_iterations": "--iter",
        "initial_model_offset_range": "--offset_range",
        "initial_model_offset_step": "--offset_step",
    }
    initial_model_flags.update(class3d_common_flags)

    initial_model_command = [
        "relion_refine",
        "--grad",
        "--denovo_3dref",
        "--i",
        particles_file,
        "--o",
        f"{job_dir.relative_to(project_dir)}/run",
        "--particle_diameter",
        f"{initial_model_params.relion_options.mask_diameter}",
        "--gpu",
        initial_model_params.initial_model_gpus,
    ]
    if initial_model_params.start_initial_model_C1:
        initial_model_command.extend(("--sym", "C1"))
    else:
        initial_model_command.extend(("--sym", initial_model_params.symmetry))
    for k, v in initial_model_params.model_dump().items():
        if k in initial_model_flags:
            if (type(v) is not bool) and (v not in [None, ""]):
                initial_model_command.extend((initial_model_flags[k], str(v)))
            elif v:
                initial_model_command.append(initial_model_flags[k])
    initial_model_command.extend(
        ("--pipeline_control", f"{job_dir.relative_to(project_dir)}/")
    )

    # Run initial model and confirm it ran successfully
    logger.info("Running initial model")
    result = subprocess.run(
        initial_model_command, cwd=str(project_dir), capture_output=True
    )

    # Register the initial model job with the node creator
    logger.info("Sending relion.initialmodel (model) to node creator")
    node_creator_parameters_refine: dict = {
        "job_type": "relion.initialmodel",
        "input_file": f"{project_dir}/{particles_file}",
        "output_file": f"{job_dir}/initial_model.mrc",
        "relion_options": dict(initial_model_params.relion_options),
        "command": " ".join(initial_model_command),
        "stdout": result.stdout.decode("utf8", "replace"),
        "stderr": result.stderr.decode("utf8", "replace"),
        "success": (result.returncode == 0),
    }
    send_to_rabbitmq("node_creator", node_creator_parameters_refine)

    # End here if the command failed
    if result.returncode:
        logger.error(
            f"Relion initial model failed with exitcode {result.returncode}:\n"
            + result.stderr.decode("utf8", "replace")
        )
        return "", []

    ini_model_file = job_dir / "initial_model.mrc"
    align_symmetry_command = [
        "relion_align_symmetry",
        "--i",
        str(
            job_dir.relative_to(project_dir)
            / f"run_it{initial_model_params.initial_model_iterations:03}_model.star"
        ),
        "--o",
        f"{ini_model_file.relative_to(project_dir)}",
        "--sym",
        initial_model_params.symmetry,
        "--apply_sym",
        "--select_largest_class",
        "--pipeline_control",
        f"{job_dir.relative_to(project_dir)}/",
    ]

    # Run symmetry alignment and confirm it ran successfully
    logger.info("Running symmetry alignment")
    result = subprocess.run(
        align_symmetry_command, cwd=str(project_dir), capture_output=True
    )

    # Register the initial model job with the node creator
    logger.info("Sending relion.initialmodel (alignment) to node creator")
    node_creator_parameters_symmetry: dict = {
        "job_type": "relion.initialmodel",
        "input_file": f"{project_dir}/{particles_file}",
        "output_file": f"{job_dir}/initial_model.mrc",
        "relion_options": dict(initial_model_params.relion_options),
        "command": " ".join(align_symmetry_command),
        "stdout": result.stdout.decode("utf8", "replace"),
        "stderr": result.stderr.decode("utf8", "replace"),
        "success": (result.returncode == 0),
    }
    send_to_rabbitmq("node_creator", node_creator_parameters_symmetry)

    # End here if the command failed
    if result.returncode:
        logger.error(
            f"Relion initial model symmetry alignment "
            f"failed with exitcode {result.returncode}:\n"
            + result.stderr.decode("utf8", "replace")
        )
        return "", []

    # Send Murfey the location of the initial model
    murfey_params = {
        "register": "save_initial_model",
        "initial_model": f"{job_dir}/initial_model.mrc",
    }
    send_to_rabbitmq("murfey_feedback", murfey_params)

    # Extract parameters for ispyb
    initial_model_cif = cif.read_file(
        f"{job_dir}/run_it{initial_model_params.initial_model_iterations:03}_model.star"
    )
    initial_model_block = initial_model_cif.find_block("model_classes")
    model_scores = list(initial_model_block.find_loop("_rlnClassDistribution"))
    model_resolutions = list(initial_model_block.find_loop("_rlnEstimatedResolution"))
    resolution = model_resolutions[model_scores == max(model_scores)]
    number_of_particles = (
        float(model_scores[model_scores == max(model_scores)])
        * initial_model_params.batch_size
    )

    logger.info(
        "Will send best initial model to ispyb with "
        f"resolution {resolution} and {number_of_particles} particles"
    )
    ini_ispyb_parameters = [
        {
            "ispyb_command": "buffer",
            "buffer_lookup": {
                "particle_classification_id": class_uuids_dict[class_uuids_keys[0]]
            },
            "buffer_command": {"ispyb_command": "insert_cryoem_initial_model"},
            "number_of_particles": number_of_particles,
            "resolution": resolution,
            "store_result": "ispyb_initial_model_id",
        }
    ]
    for i in range(1, initial_model_params.class3d_nr_classes):
        # Insert initial model for every class, sending model id each time
        ini_ispyb_parameters.append(
            {
                "ispyb_command": "buffer",
                "buffer_lookup": {
                    "particle_classification_id": class_uuids_dict[class_uuids_keys[i]]
                },
                "buffer_command": {"ispyb_command": "insert_cryoem_initial_model"},
                "number_of_particles": number_of_particles,
                "resolution": resolution,
                "cryoem_initial_model_id": "$ispyb_initial_model_id",
            }
        )
    for i in range(initial_model_params.class3d_nr_classes):
        # Add resolution to every model if it is finite
        if np.isfinite(float(resolution)):
            ini_ispyb_parameters[i]["resolution"] = resolution
        else:
            ini_ispyb_parameters[i]["resolution"] = 0.0

    logger.info("Running 3D classification using new initial model")
    return f"{ini_model_file}", ini_ispyb_parameters


def run_class3d(class3d_params: Class3DParameters, send_to_rabbitmq: Callable) -> bool:
    # Class ids get fed in as a string, need to convert these to a dictionary
    class_uuids_dict = json.loads(class3d_params.class_uuids.replace("'", '"'))
    class_uuids_keys = list(class_uuids_dict.keys())

    # Update the relion options to get out the mask diameter
    class3d_params.relion_options = update_relion_options(
        class3d_params.relion_options, dict(class3d_params)
    )

    # Make the job directory and move to the project directory
    job_dir = Path(class3d_params.class3d_dir)
    if (job_dir / "RELION_JOB_EXIT_SUCCESS").exists():
        # This job over-writes a previous one
        job_is_rerun = True
    else:
        job_is_rerun = False
        job_dir.mkdir(parents=True, exist_ok=True)
    project_dir = job_dir.parent.parent
    os.chdir(project_dir)

    particles_file = str(Path(class3d_params.particles_file).relative_to(project_dir))
    logger.info(f"Running Class3D for {particles_file}")

    # Run the initial model if requested, otherwise look for a pre-existing file
    if class3d_params.do_initial_model:
        job_num_search = re.search("/job[0-9]+", class3d_params.class3d_dir)
        if job_num_search:
            job_num_3d = int(job_num_search[0][4:7])
        else:
            logger.warning(f"Invalid job directory in {class3d_params.class3d_dir}")
            return False
        initial_model_file, initial_model_ispyb_parameters = run_initial_model(
            initial_model_params=class3d_params,
            project_dir=project_dir,
            job_num=job_num_3d - 1,
            send_to_rabbitmq=send_to_rabbitmq,
        )
    else:
        initial_model_file = class3d_params.initial_model_file or ""
        initial_model_ispyb_parameters = []
    if not initial_model_file:
        # If there isn't an initial model file something has gone wrong
        logger.error("No initial model file found, stopping.")
        return False

    class3d_flags = {
        "dont_correct_greyscale": "--firstiter_cc",
        "initial_lowpass": "--ini_high",
        "class3d_nr_iter": "--iter",
        "fast_subsets": "--fast_subsets",
        "tau_fudge": "--tau2_fudge",
        "highres_limit": "--strict_highres_exp",
        "fn_mask": "--solvent_mask",
        "skip_align": "--skip_align",
        "offset_range": "--offset_range",
        "offset_step": "--offset_step",
        "allow_coarser": "--allow_coarser_sampling",
        "symmetry": "--sym",
        "do_norm": "--norm",
        "do_scale": "--scale",
        "gpus": "--gpu",
    }
    class3d_flags.update(class3d_common_flags)

    # Create the classification command
    class3d_command = class3d_params.mpi_run_command.split()
    class3d_command.extend(
        [
            "relion_refine_mpi",
            "--i",
            particles_file,
            "--o",
            f"{job_dir.relative_to(project_dir)}/run",
            "--ref",
            initial_model_file,
            "--particle_diameter",
            f"{class3d_params.relion_options.mask_diameter}",
        ]
    )
    for k, v in class3d_params.model_dump().items():
        if k in class3d_flags:
            if (type(v) is not bool) and (v not in [None, ""]):
                class3d_command.extend((class3d_flags[k], str(v)))
            elif v:
                class3d_command.append(class3d_flags[k])
    class3d_command.extend(
        ("--pipeline_control", f"{job_dir.relative_to(project_dir)}/")
    )

    # Run Class3D and confirm it ran successfully
    logger.info(" ".join(class3d_command))
    result = subprocess.run(class3d_command, cwd=str(project_dir), capture_output=True)
    if not job_is_rerun:
        (job_dir / "RELION_JOB_EXIT_SUCCESS").unlink(missing_ok=True)

    # Register the Class3D job with the node creator
    logger.info(f"Sending {job_type} to node creator")
    node_creator_parameters: dict = {
        "job_type": job_type,
        "input_file": class3d_params.particles_file + f":{initial_model_file}",
        "output_file": class3d_params.class3d_dir,
        "relion_options": dict(class3d_params.relion_options),
        "command": " ".join(class3d_command),
        "stdout": result.stdout.decode("utf8", "replace"),
        "stderr": result.stderr.decode("utf8", "replace"),
        "success": (result.returncode == 0),
    }
    send_to_rabbitmq("node_creator", node_creator_parameters)

    # End here if the command failed
    if result.returncode:
        logger.error(
            f"Relion Class3D failed with exitcode {result.returncode}:\n"
            + result.stderr.decode("utf8", "replace")
        )
        return False

    # Generate healpix image of the particle distribution
    logger.info("Generating healpix angular distribution image")
    data = cif.read_file(
        str(job_dir / f"run_it{class3d_params.class3d_nr_iter:03}_data.star")
    )
    optics_block = data.find_block("optics")
    binned_pixel_size = optics_block.find_loop("_rlnImagePixelSize")[0]

    particles_block = data.find_block("particles")
    class_efficiencies = np.zeros(class3d_params.class3d_nr_classes)
    if particles_block:
        angles_rot = np.array(particles_block.find_loop("_rlnAngleRot"), dtype=float)
        angles_tilt = np.array(particles_block.find_loop("_rlnAngleTilt"), dtype=float)
        class_numbers = np.array(
            particles_block.find_loop("_rlnClassNumber"), dtype=int
        )

        for class_id in range(class3d_params.class3d_nr_classes):
            if not len(angles_tilt[class_numbers == class_id + 1]):
                # Skip any classes with no particles
                continue

            try:
                angular_distribution_plot(
                    theta_degrees=angles_tilt[class_numbers == class_id + 1],
                    phi_degrees=angles_rot[class_numbers == class_id + 1],
                    healpix_order=class3d_params.healpix_order,
                    output_jpeg=job_dir
                    / f"run_it{class3d_params.class3d_nr_iter:03}_class{class_id + 1:03}_angdist.jpeg",
                    class_label=str(class_id + 1),
                )
            except ValueError as e:
                logger.error(f"Angular distribution plotting failed: {e}")

            class_efficiencies[class_id] = find_efficiency(
                theta_degrees=angles_tilt[class_numbers == class_id + 1],
                phi_degrees=angles_rot[class_numbers == class_id + 1],
            )
            logger.info(
                f"Efficiency of class {class_id + 1} is {class_efficiencies[class_id]}"
            )

    # Send classification job information to ispyb
    ispyb_parameters = []
    classification_grp_ispyb_parameters = {
        "ispyb_command": "buffer",
        "buffer_command": {"ispyb_command": "insert_particle_classification_group"},
        "type": "3D",
        "batch_number": "1",
        "number_of_particles_per_batch": class3d_params.batch_size,
        "number_of_classes_per_batch": class3d_params.class3d_nr_classes,
        "symmetry": class3d_params.symmetry,
        "binned_pixel_size": binned_pixel_size,
        "particle_picker_id": class3d_params.picker_id,
    }
    if job_is_rerun:
        # If this job overwrites another get the id for it
        classification_grp_ispyb_parameters["buffer_lookup"] = {
            "particle_classification_group_id": class3d_params.class3d_grp_uuid,
        }
    else:
        classification_grp_ispyb_parameters["buffer_store"] = (
            class3d_params.class3d_grp_uuid
        )
    ispyb_parameters.append(classification_grp_ispyb_parameters)

    # Send individual classes to ispyb
    class_star_file = cif.read_file(
        f"{class3d_params.class3d_dir}/run_it{class3d_params.class3d_nr_iter:03}_model.star"
    )
    classes_block = class_star_file.find_block("model_classes")
    classes_loop = classes_block.find_loop("_rlnReferenceImage").get_loop()

    class_ids = []
    class_resolutions = []
    class_completenesses = []
    class_sort_criteria = []

    for class_id in range(class3d_params.class3d_nr_classes):
        # Add an ispyb insert for each class
        class_ispyb_parameters = {
            "ispyb_command": "buffer",
            "buffer_lookup": {
                "particle_classification_group_id": class3d_params.class3d_grp_uuid
            },
            "buffer_command": {"ispyb_command": "insert_particle_classification"},
            "class_number": class_id + 1,
            "class_image_full_path": (
                f"{class3d_params.class3d_dir}/"
                f"run_it{class3d_params.class3d_nr_iter:03}_class{class_id + 1:03}.mrc"
            ),
            "particles_per_class": (
                float(classes_loop[class_id, 1]) * class3d_params.batch_size
            ),
            "class_distribution": classes_loop[class_id, 1],
            "rotation_accuracy": classes_loop[class_id, 2],
            "translation_accuracy": classes_loop[class_id, 3],
            "angular_efficiency": class_efficiencies[class_id],
            "suggested_tilt": 0 if class_efficiencies[class_id] > 0.65 else 30,
        }
        if job_is_rerun:
            class_ispyb_parameters["buffer_lookup"].update(
                {
                    "particle_classification_id": class_uuids_dict[
                        class_uuids_keys[class_id]
                    ],
                }
            )
        else:
            class_ispyb_parameters["buffer_store"] = class_uuids_dict[
                class_uuids_keys[class_id]
            ]

        # Add the resolution and fourier completeness if they are valid numbers
        estimated_resolution = float(classes_loop[class_id, 4])
        if np.isfinite(estimated_resolution):
            class_ispyb_parameters["estimated_resolution"] = estimated_resolution
        else:
            class_ispyb_parameters["estimated_resolution"] = 99.0
        fourier_completeness = float(classes_loop[class_id, 5])
        if np.isfinite(fourier_completeness):
            class_ispyb_parameters["overall_fourier_completeness"] = (
                fourier_completeness
            )
        else:
            class_ispyb_parameters["overall_fourier_completeness"] = 0.0

        # Add this class to the list of resolutions
        class_ids.append(class_id + 1)
        class_resolutions.append(class_ispyb_parameters["estimated_resolution"])
        class_completenesses.append(
            class_ispyb_parameters["overall_fourier_completeness"]
        )

        # Sorting criteria are resolution, descending efficiency and reversed particle count
        class_sort_criteria.append(
            (
                class_ispyb_parameters["estimated_resolution"],
                1 - class_efficiencies[class_id],
                class_ispyb_parameters["particles_per_class"]
                - class3d_params.batch_size,
            )
        )

        # Add the ispyb command to the command list
        ispyb_parameters.append(class_ispyb_parameters)

    # Add on the initial model insert before sending
    if class3d_params.do_initial_model:
        ispyb_parameters.extend(initial_model_ispyb_parameters)

    logger.info(f"Sending to ispyb {ispyb_parameters}")
    send_to_rabbitmq(
        "ispyb_connector",
        {
            "ispyb_command": "multipart_message",
            "ispyb_command_list": ispyb_parameters,
        },
    )

    # Prepare a message to tell Murfey this wrapper is done
    murfey_params = {
        "register": "done_3d_batch",
        "refine_dir": f"{project_dir}/Refine3D/job",
        "class3d_dir": class3d_params.class3d_dir,
        "best_class": 0,
        "do_refinement": False,
    }

    # Work out the best class and request refinement if it meets the target criteria
    class_sorting_array = np.array(
        class_sort_criteria,
        dtype=[("resolutions", "<f"), ("efficiencies", "<f"), ("particles", "<i")],
    )
    class_sorting = np.argsort(
        class_sorting_array, order=("resolutions", "efficiencies", "particles")
    )
    for cid in class_sorting:
        if (
            class3d_params.batch_size == 200000
            and class_resolutions[cid] < 11
            and (class_efficiencies[cid] > 0.65 or class3d_params.symmetry != "C1")
        ):
            murfey_params["do_refinement"] = True
            murfey_params["best_class"] = class_ids[cid]
            break

    # Tell Murfey the batch has finished
    send_to_rabbitmq("murfey_feedback", murfey_params)

    (job_dir / "RELION_JOB_EXIT_SUCCESS").touch(exist_ok=True)
    logger.info(f"Done {job_type} for {class3d_params.particles_file}.")
    return True


class Class3DWrapper:
    """
    A wrapper for the Relion 3D classification job.
    """

    def __init__(self, recwrap):
        self.log = logging.LoggerAdapter(logger)
        self.recwrap = recwrap
        self.class_uuids_dict = {}
        self.class_uuids_keys = []

    def run(self):
        """
        Run the 3D classification and register results
        """
        params_dict = self.recwrap.recipe_step["job_parameters"]
        try:
            class3d_params = Class3DParameters(**params_dict)
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Class3D parameter validation failed for parameters: {params_dict} "
                f"with exception: {e}"
            )
            return False

        successful_run = run_class3d(
            class3d_params, send_to_rabbitmq=self.recwrap.send_to
        )
        if not successful_run:
            return False
        return True
