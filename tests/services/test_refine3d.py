from __future__ import annotations

import copy
from pathlib import Path
from unittest import mock

import pytest
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.refine3d import Refine3D
from cryoemservices.util.relion_service_options import RelionServiceOptions


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "ack")
    mocker.spy(transport, "nack")
    return transport


@mock.patch("cryoemservices.wrappers.refine3d_wrapper.subprocess.run")
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
def test_refine3d_service_with_mask(
    mock_recwrap_send, mock_subprocess, offline_transport, tmp_path
):
    """
    Send a test message to the Refine3D service with a provided mask.
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
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    refine3d_test_message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "parameters": {
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
            },
        },
        "recipe-pointer": 1,
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

    # Set up and run the service
    service = Refine3D(environment={"queue": ""}, rabbitmq_credentials=Path("."))
    service._transport = offline_transport
    service.initializing()
    service.refine3d(rw=recipe_wrapper, header=header, message={})

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


@mock.patch("cryoemservices.services.refine3d.run_refinement")
def test_refine3d_service_failed_resends(mock_refine3d, offline_transport, tmp_path):
    """Failures of the processing should lead to reinjection of the message"""

    def raise_exception(*args, **kwargs):
        raise ValueError

    mock_refine3d.side_effect = raise_exception

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    refine_test_message = {
        "batch_size": 50000,
        "class_number": 1,
        "is_first_refinement": True,
        "mask_diameter": "190.0",
        "number_of_particles": 10000,
        "particle_diameter": "180",
        "particles_file": f"{tmp_path}/Extract/job020/particles.star",
        "pixel_size": "3.85",
        "refine_job_dir": f"{tmp_path}/Refine3D/job021",
        "relion_options": {},
        "rescaled_class_reference": f"{tmp_path}/Extract/job020/ref.mrc",
    }
    end_message = copy.deepcopy(refine_test_message)

    # Set up and run the service
    service = Refine3D(environment={"queue": ""}, rabbitmq_credentials=Path("."))
    service._transport = offline_transport
    service.initializing()
    service.refine3d(None, header=header, message=refine_test_message)

    end_message["requeue"] = 1
    offline_transport.send.assert_any_call("refine3d", end_message)
    offline_transport.ack.assert_called_once()


def test_refine3d_service_nack_on_requeue(offline_transport, tmp_path):
    """Messages reinjected 5 times should nack"""
    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    refine_test_message = {
        "batch_size": 50000,
        "class_number": 1,
        "is_first_refinement": True,
        "mask_diameter": "190.0",
        "number_of_particles": 10000,
        "particle_diameter": "180",
        "particles_file": f"{tmp_path}/Extract/job020/particles.star",
        "pixel_size": "3.85",
        "refine_job_dir": f"{tmp_path}/Refine3D/job021",
        "relion_options": {},
        "rescaled_class_reference": f"{tmp_path}/Extract/job020/ref.mrc",
        "requeue": 5,
    }

    # Set up and run the service
    service = Refine3D(environment={"queue": ""}, rabbitmq_credentials=Path("."))
    service._transport = offline_transport
    service.initializing()
    service.refine3d(None, header=header, message=refine_test_message)

    assert offline_transport.send.call_count == 0
    offline_transport.nack.assert_called_once()


def test_refine3d_service_nack_wrong_params(offline_transport, tmp_path):
    """Messages without required parameters should nack"""
    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    refine_test_message = {
        "relion_options": {},
    }

    # Set up and run the service
    service = Refine3D(environment={"queue": ""}, rabbitmq_credentials=Path("."))
    service._transport = offline_transport
    service.initializing()
    service.refine3d(None, header=header, message=refine_test_message)

    assert offline_transport.send.call_count == 0
    offline_transport.nack.assert_called_once()
