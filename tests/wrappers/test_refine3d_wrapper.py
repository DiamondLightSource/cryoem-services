from __future__ import annotations

import sys
from unittest import mock

import mrcfile
import numpy as np
import pytest
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.util.relion_service_options import RelionServiceOptions
from cryoemservices.wrappers import refine3d_wrapper


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.wrappers.refine3d_wrapper.subprocess.run")
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
def test_refine3d_wrapper_with_mask(
    mock_recwrap_send, mock_subprocess, offline_transport, tmp_path
):
    """
    Send a test message to the Refine3D wrapper with a provided mask.
    The 3D refinment command should be run,
    then cause node_creator and postprocessing messages.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("utf8")
    mock_subprocess().stderr = "stderr".encode("utf8")

    rescaling_command = [
        "relion_image_handler",
        "--i",
        f"{tmp_path}/Class3D/job015/ref.mrc",
        "--o",
        f"{tmp_path}/Extract/job020/ref.mrc",
        "--angpix",
        "4.1",
        "--rescale_angpix",
        "3.85",
        "--force_header_angpix",
        "3.85",
        "--new_box",
        "200",
    ]

    # Example recipe wrapper message to run the service with a few parameters varied
    refine3d_test_message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "job_parameters": {
                    "batch_size": 50000,
                    "class_number": 1,
                    "ctf_intact_first_peak": False,
                    "do_ctf": True,
                    "do_norm": True,
                    "do_scale": True,
                    "do_zero_mask": True,
                    "dont_combine_weights_via_disc": True,
                    "dont_correct_greyscale": True,
                    "flattern_solvent": True,
                    "gpus": "0",
                    "healpix_order": 2,
                    "ignore_angles": True,
                    "initial_lowpass": "40.0",
                    "is_first_refinement": False,
                    "local_healpix_order": 4,
                    "low_resol_join_halves": 30,
                    "mask": f"{tmp_path}/MaskCreate/job019/mask.mrc",
                    "mpi_run_command": "srun -n 9",
                    "nr_pool": 5,
                    "number_of_particles": 10000,
                    "pad": 2,
                    "particle_diameter": "180",
                    "particles_file": f"{tmp_path}/Extract/job020/particles.star",
                    "pixel_size": "3.85",
                    "preread_images": False,
                    "offset_range": 5,
                    "offset_step": 2,
                    "oversampling": 1,
                    "refine_job_dir": f"{tmp_path}/Refine3D/job021",
                    "relion_options": {},
                    "rescaled_class_reference": f"{tmp_path}/Extract/job020/ref.mrc",
                    "rescaling_command": rescaling_command,
                    "resol_angles": True,
                    "scratch_dir": None,
                    "symmetry": "C3",
                    "threads": 4,
                },
                "parameters": {
                    "cluster": {
                        "gpus": 4,
                        "tasks": 9,
                    },
                    "recipewrapper": f"{tmp_path}/Refine3D/job021/.recipewrap",
                    "workingdir": f"{tmp_path}/Refine3D/job021/",
                },
                "queue": "cluster.submission",
                "service": "Refine3DWrapper",
                "wrapper": {"task_information": "Refine3D"},
            },
        },
        "recipe-pointer": 1,
        "environment": {"ID": "envID"},
        "recipe-path": [],
        "payload": [],
    }
    output_relion_options = RelionServiceOptions()
    output_relion_options.particle_diameter = 180
    output_relion_options.initial_lowpass = 40.0
    output_relion_options.pixel_size = 3.85
    output_relion_options.symmetry = "C3"
    output_relion_options = dict(output_relion_options)

    # Create the expected output files
    (tmp_path / "Extract/job020").mkdir(parents=True)
    (tmp_path / "Extract/job020/ref.mrc").touch()

    (tmp_path / "Refine3D/job021").mkdir(parents=True)
    with open(tmp_path / "Refine3D/job021/run_data.star", "w") as data_star:
        data_star.write(
            "data_particles\nloop_\n_rlnAngleRot\n_rlnAngleTilt\n_rlnClassNumber\n"
            "0.5 1.0 1\n1.5 2.0 1\n2.5 3.0 2\n3.5 4.0 2\n"
        )

    # Create a recipe wrapper with the test message
    recipe_wrapper = RecipeWrapper(
        message=refine3d_test_message, transport=offline_transport
    )

    # Set up and run the mock service
    service_wrapper = refine3d_wrapper.Refine3DWrapper()
    service_wrapper.set_recipe_wrapper(recipe_wrapper)
    service_wrapper.run()

    # Check the expected refinement command was run
    assert mock_subprocess.call_count == 5
    mock_subprocess.assert_any_call(
        rescaling_command, capture_output=True, cwd=str(tmp_path)
    )

    refine3d_command = [
        "srun",
        "-n",
        "9",
        "relion_refine_mpi",
        "--i",
        f"{tmp_path}/Extract/job020/particles.star",
        "--o",
        f"{tmp_path}/Refine3D/job021/run",
        "--ref",
        f"{tmp_path}/Extract/job020/ref.mrc",
        "--particle_diameter",
        str(output_relion_options["mask_diameter"]),
        "--auto_refine",
        "--split_random_halves",
        "--firstiter_cc",
        "--ini_high",
        "40.0",
        "--dont_combine_weights_via_disc",
        "--pool",
        "5",
        "--pad",
        "2",
        "--ctf",
        "--flatten_solvent",
        "--zero_mask",
        "--oversampling",
        "1",
        "--healpix_order",
        "2",
        "--auto_local_healpix_order",
        "4",
        "--low_resol_join_halves",
        "30.0",
        "--offset_range",
        "5.0",
        "--offset_step",
        "2.0",
        "--auto_ignore_angles",
        "--auto_resol_angles",
        "--sym",
        str(output_relion_options["symmetry"]),
        "--norm",
        "--scale",
        "--j",
        "4",
        "--gpu",
        "0",
        "--pipeline_control",
        f"{tmp_path}/Refine3D/job021/",
    ]
    mock_subprocess.assert_any_call(
        refine3d_command, capture_output=True, cwd=str(tmp_path)
    )

    # Check the expected message sends to other processes
    assert mock_recwrap_send.call_count == 2
    mock_recwrap_send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.refine3d",
            "input_file": f"{tmp_path}/Extract/job020/particles.star:{tmp_path}/Extract/job020/ref.mrc",
            "output_file": f"{tmp_path}/Refine3D/job021/",
            "relion_options": output_relion_options,
            "command": " ".join(refine3d_command),
            "stdout": "stdout",
            "stderr": "stderr",
            "alias": "Refine_C3_symmetry",
            "success": True,
        },
    )
    mock_recwrap_send.assert_any_call(
        "postprocess",
        {
            "half_map": f"{tmp_path}/Refine3D/job021/run_half1_class001_unfil.mrc",
            "mask": f"{tmp_path}/MaskCreate/job019/mask.mrc",
            "rescaled_class_reference": f"{tmp_path}/Extract/job020/ref.mrc",
            "job_dir": f"{tmp_path}/PostProcess/job022",
            "is_first_refinement": False,
            "pixel_size": 3.85,
            "number_of_particles": 10000,
            "batch_size": 50000,
            "class_number": 1,
            "symmetry": "C3",
            "particles_file": f"{tmp_path}/Extract/job020/particles.star",
            "relion_options": output_relion_options,
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.wrappers.refine3d_wrapper.subprocess.run")
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
@mock.patch("cryoemservices.wrappers.refine3d_wrapper.find_mask_threshold")
def test_refine3d_wrapper_no_mask(
    mock_mask_threshold,
    mock_recwrap_send,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to the Refine3D wrapper without a mask.
    The 3D refinement and mask creation commands should be run,
    then cause node_creator and postprocessing messages.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("utf8")
    mock_subprocess().stderr = "stderr".encode("utf8")

    mock_mask_threshold.return_value = 0.1

    rescaling_command = [
        "relion_image_handler",
        "--i",
        f"{tmp_path}/Class3D/job015/ref.mrc",
        "--o",
        f"{tmp_path}/Extract/job020/ref.mrc",
        "--angpix",
        "4.1",
        "--rescale_angpix",
        "3.85",
        "--force_header_angpix",
        "3.85",
        "--new_box",
        "200",
    ]

    # Example recipe wrapper message to run the service with a few parameters varied
    refine3d_test_message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "job_parameters": {
                    "batch_size": 50000,
                    "class_number": 1,
                    "is_first_refinement": True,
                    "mask_diameter": "190.0",
                    "mask_extend": 2,
                    "mask_lowpass": 14,
                    "mask_soft_edge": 2,
                    "number_of_particles": 10000,
                    "particle_diameter": "180",
                    "particles_file": f"{tmp_path}/Extract/job020/particles.star",
                    "pixel_size": "3.85",
                    "refine_job_dir": f"{tmp_path}/Refine3D/job021",
                    "relion_options": {},
                    "rescaled_class_reference": f"{tmp_path}/Extract/job020/ref.mrc",
                    "rescaling_command": rescaling_command,
                },
                "parameters": {
                    "cluster": {
                        "gpus": 4,
                        "tasks": 9,
                    },
                    "recipewrapper": f"{tmp_path}/Refine3D/job021/.recipewrap",
                    "workingdir": f"{tmp_path}/Refine3D/job021/",
                },
                "queue": "cluster.submission",
                "service": "Refine3DWrapper",
                "wrapper": {"task_information": "Refine3D"},
            },
        },
        "recipe-pointer": 1,
        "environment": {"ID": "envID"},
        "recipe-path": [],
        "payload": [],
    }
    output_relion_options = RelionServiceOptions()
    output_relion_options.initial_lowpass = 20.0
    output_relion_options.mask_lowpass = 14.0

    output_relion_options.particle_diameter = 180
    output_relion_options.pixel_size = 3.85
    output_relion_options = dict(output_relion_options)

    # Create the expected output files
    (tmp_path / "Extract/job020").mkdir(parents=True)
    (tmp_path / "Extract/job020/ref.mrc").touch()

    (tmp_path / "Refine3D/job021").mkdir(parents=True)
    (tmp_path / "Refine3D/job021/run_class001.mrc").touch()
    with open(tmp_path / "Refine3D/job021/run_data.star", "w") as data_star:
        data_star.write(
            "data_particles\nloop_\n_rlnAngleRot\n_rlnAngleTilt\n_rlnClassNumber\n"
            "0.5 1.0 1\n1.5 2.0 1\n2.5 3.0 2\n3.5 4.0 2\n"
        )

    # Create a recipe wrapper with the test message
    recipe_wrapper = RecipeWrapper(
        message=refine3d_test_message, transport=offline_transport
    )

    # Set up and run the mock service
    service_wrapper = refine3d_wrapper.Refine3DWrapper()
    service_wrapper.set_recipe_wrapper(recipe_wrapper)
    service_wrapper.run()

    # Check the expected refinement command was run
    assert mock_subprocess.call_count == 6
    mock_subprocess.assert_any_call(
        rescaling_command, capture_output=True, cwd=str(tmp_path)
    )

    refine3d_command = [
        "srun",
        "-n",
        "5",
        "relion_refine_mpi",
        "--i",
        f"{tmp_path}/Extract/job020/particles.star",
        "--o",
        f"{tmp_path}/Refine3D/job021/run",
        "--ref",
        f"{tmp_path}/Extract/job020/ref.mrc",
        "--particle_diameter",
        str(output_relion_options["mask_diameter"]),
        "--auto_refine",
        "--split_random_halves",
        "--firstiter_cc",
        "--ini_high",
        "20.0",
        "--dont_combine_weights_via_disc",
        "--preread_images",
        "--pool",
        "10",
        "--pad",
        "2",
        "--ctf",
        "--flatten_solvent",
        "--zero_mask",
        "--oversampling",
        "1",
        "--healpix_order",
        "2",
        "--auto_local_healpix_order",
        "4",
        "--low_resol_join_halves",
        "40",
        "--offset_range",
        "5",
        "--offset_step",
        "4",
        "--sym",
        str(output_relion_options["symmetry"]),
        "--norm",
        "--scale",
        "--j",
        "8",
        "--gpu",
        "0:1:2:3",
        "--pipeline_control",
        f"{tmp_path}/Refine3D/job021/",
    ]
    mock_subprocess.assert_any_call(
        refine3d_command, capture_output=True, cwd=str(tmp_path)
    )

    mask_creation_command = [
        "relion_mask_create",
        "--i",
        f"{tmp_path}/Refine3D/job021/run_class001.mrc",
        "--o",
        "MaskCreate/job022/mask.mrc",
        "--lowpass",
        "14.0",
        "--ini_threshold",
        "0.1",
        "--extend_inimask",
        "2",
        "--width_soft_edge",
        "2",
        "--angpix",
        "3.85",
        "--j",
        "8",
        "--pipeline_control",
        "MaskCreate/job022/",
    ]
    mock_subprocess.assert_any_call(
        mask_creation_command, capture_output=True, cwd=str(tmp_path)
    )

    # Check the expected message sends to other processes
    assert mock_recwrap_send.call_count == 3
    mock_recwrap_send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.refine3d",
            "input_file": f"{tmp_path}/Extract/job020/particles.star:{tmp_path}/Extract/job020/ref.mrc",
            "output_file": f"{tmp_path}/Refine3D/job021/",
            "relion_options": output_relion_options,
            "command": " ".join(refine3d_command),
            "stdout": "stdout",
            "stderr": "stderr",
            "alias": "Refine_C1_symmetry",
            "success": True,
        },
    )
    output_relion_options["mask_threshold"] = 0.1
    mock_recwrap_send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.maskcreate",
            "input_file": f"{tmp_path}/Refine3D/job021/run_class001.mrc",
            "output_file": f"{tmp_path}/MaskCreate/job022/mask.mrc",
            "relion_options": output_relion_options,
            "command": " ".join(mask_creation_command),
            "stdout": "stdout",
            "stderr": "stderr",
            "alias": "Mask_C1_symmetry",
            "success": True,
        },
    )
    mock_recwrap_send.assert_any_call(
        "postprocess",
        {
            "half_map": f"{tmp_path}/Refine3D/job021/run_half1_class001_unfil.mrc",
            "mask": f"{tmp_path}/MaskCreate/job022/mask.mrc",
            "rescaled_class_reference": f"{tmp_path}/Extract/job020/ref.mrc",
            "job_dir": f"{tmp_path}/PostProcess/job023",
            "is_first_refinement": True,
            "pixel_size": 3.85,
            "number_of_particles": 10000,
            "batch_size": 50000,
            "class_number": 1,
            "symmetry": "C1",
            "particles_file": f"{tmp_path}/Extract/job020/particles.star",
            "relion_options": output_relion_options,
        },
    )


def test_find_mask_threshold_noise(tmp_path):
    """
    Test using a flat input.
    Returns the max of the binning due to the length of the array
    """
    test_matrix = np.arange(-0.0999, 0.1001, 0.0002, dtype=np.float32).reshape(
        (10, 10, 10)
    )
    with mrcfile.new(tmp_path / "test_density_file.mrc") as mrc:
        mrc.set_data(test_matrix)
    found_density = refine3d_wrapper.find_mask_threshold(
        f"{tmp_path}/test_density_file.mrc"
    )
    assert np.abs(found_density - 0.1) < 0.0001


def test_find_mask_threshold_sample(tmp_path):
    """
    Test using a flat input, plus some extra values of 0.05 which should be the peak
    Returns the max of the binning due to the length of the array
    """
    test_matrix = np.arange(-0.0999, 0.1001, 0.0002, dtype=np.float32).reshape(
        (10, 10, 10)
    )
    test_matrix[0, 0, 0] = 0.05
    test_matrix[-1, -1, -1] = 0.05
    with mrcfile.new(tmp_path / "test_density_file.mrc") as mrc:
        mrc.set_data(test_matrix)
    found_density = refine3d_wrapper.find_mask_threshold(
        f"{tmp_path}/test_density_file.mrc"
    )
    assert np.abs(found_density - 0.05) < 0.0001
