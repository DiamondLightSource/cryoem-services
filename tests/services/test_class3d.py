from __future__ import annotations

import copy
from pathlib import Path
from unittest import mock

import pytest
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.class3d import Class3D
from cryoemservices.util.relion_service_options import RelionServiceOptions


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "ack")
    mocker.spy(transport, "nack")
    return transport


@mock.patch("cryoemservices.wrappers.class3d_wrapper.find_efficiency")
@mock.patch("cryoemservices.wrappers.class3d_wrapper.subprocess.run")
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
def test_class3d_service_has_initial_model(
    mock_recwrap_send, mock_subprocess, mock_efficiency, offline_transport, tmp_path
):
    """
    Send a test message to the Class3D service for a second round of 100000 particles,
    with a provided initial model.
    The 3D classification command should be run,
    then cause ispyb, node_creator and murfey messages.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("utf8")
    mock_subprocess().stderr = "stderr".encode("utf8")
    mock_efficiency.return_value = 0.6

    # Example recipe wrapper message to run the service with a few parameters varied
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    class3d_test_message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "parameters": {
                    "batch_size": "100000",
                    "class_uuids": "{'0': 10, '1': 11}",
                    "class3d_dir": f"{tmp_path}/Class3D/job015",
                    "class3d_grp_uuid": "5",
                    "class3d_nr_classes": "2",
                    "do_initial_model": False,
                    "initial_model_file": f"{tmp_path}/initial_model.mrc",
                    "mask_diameter": "190.0",
                    "particle_diameter": "180",
                    "particles_file": f"{tmp_path}/Select/job013/particles_100000.star",
                    "picker_id": "6",
                    "relion_options": {},
                },
            },
        },
        "recipe-pointer": 1,
    }
    output_relion_options = RelionServiceOptions()
    output_relion_options.particle_diameter = 180
    output_relion_options.class3d_nr_classes = 2
    output_relion_options.batch_size = 100000
    output_relion_options.class3d_nr_iter = 20
    output_relion_options = dict(output_relion_options)

    # Create the expected output files
    (tmp_path / "initial_model.mrc").touch()

    (tmp_path / "Class3D/job015").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "Class3D/job015/run_it020_data.star", "w") as data_star:
        data_star.write(
            "data_optics\nloop_\n_rlnImagePixelSize\n2.5\n\n"
            "data_particles\nloop_\n_rlnAngleRot\n_rlnAngleTilt\n_rlnClassNumber\n"
            "0.5 1.0 1\n1.5 2.0 1\n2.5 3.0 2\n3.5 4.0 2\n"
        )
    with open(tmp_path / "Class3D/job015/run_it020_model.star", "w") as model_star:
        model_star.write(
            "data_model_classes\nloop_\n"
            "_rlnReferenceImage\n_Fraction\n_Rotation\n_Translation\n"
            "_Resolution\n_Completeness\n_OffsetX\n_OffsetY\n"
            "1@Class3D/job015/run_it020_classes.mrcs 0.4 30.3 33.3 12.2 1.0 0.6 0.01\n"
            "2@Class3D/job015/run_it020_classes.mrcs 0.6 20.2 22.2 10.0 0.9 -0.5 -0.02"
        )

    # Create a recipe wrapper with the test message
    recipe_wrapper = RecipeWrapper(
        message=class3d_test_message, transport=offline_transport
    )

    # Set up and run the service
    service = Class3D(environment={"queue": ""}, rabbitmq_credentials=Path("."))
    service._transport = offline_transport
    service.initializing()
    service.class3d(rw=recipe_wrapper, header=header, message={})

    # Check the expected 3D classifcation command was run
    assert mock_subprocess.call_count == 4
    class3d_command = [
        "srun",
        "-n",
        "5",
        "relion_refine_mpi",
        "--i",
        "Select/job013/particles_100000.star",
        "--o",
        "Class3D/job015/run",
        "--ref",
        f"{tmp_path}/initial_model.mrc",
        "--particle_diameter",
        str(output_relion_options["mask_diameter"]),
        "--dont_combine_weights_via_disc",
        "--preread_images",
        "--pool",
        "10",
        "--pad",
        "2",
        "--firstiter_cc",
        "--ini_high",
        "40.0",
        "--ctf",
        "--iter",
        "20",
        "--tau2_fudge",
        "4",
        "--K",
        "2",
        "--flatten_solvent",
        "--zero_mask",
        "--oversampling",
        "1",
        "--healpix_order",
        "2",
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
        "Class3D/job015/",
    ]
    mock_subprocess.assert_any_call(
        class3d_command, capture_output=True, cwd=str(tmp_path)
    )

    # Check the expected message sends to other processes
    assert mock_recwrap_send.call_count == 3
    mock_recwrap_send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.class3d",
            "input_file": f"{tmp_path}/Select/job013/particles_100000.star:{tmp_path}/initial_model.mrc",
            "output_file": f"{tmp_path}/Class3D/job015",
            "relion_options": output_relion_options,
            "command": " ".join(class3d_command),
            "stdout": "stdout",
            "stderr": "stderr",
            "success": True,
        },
    )
    mock_recwrap_send.assert_any_call(
        "ispyb_connector",
        {
            "ispyb_command": "multipart_message",
            "ispyb_command_list": [
                {
                    "batch_number": "1",
                    "binned_pixel_size": "2.5",
                    "buffer_command": {
                        "ispyb_command": "insert_particle_classification_group"
                    },
                    "buffer_store": 5,
                    "ispyb_command": "buffer",
                    "number_of_classes_per_batch": 2,
                    "number_of_particles_per_batch": 100000,
                    "particle_picker_id": 6,
                    "symmetry": str(output_relion_options["symmetry"]),
                    "type": "3D",
                },
                {
                    "buffer_command": {
                        "ispyb_command": "insert_particle_classification"
                    },
                    "buffer_lookup": {"particle_classification_group_id": 5},
                    "buffer_store": 10,
                    "class_distribution": "0.4",
                    "class_image_full_path": (
                        f"{tmp_path}/Class3D/job015/run_it020_class001.mrc"
                    ),
                    "class_number": 1,
                    "estimated_resolution": 12.2,
                    "ispyb_command": "buffer",
                    "overall_fourier_completeness": 1.0,
                    "particles_per_class": 40000.0,
                    "rotation_accuracy": "30.3",
                    "translation_accuracy": "33.3",
                    "angular_efficiency": 0.6,
                    "suggested_tilt": 30,
                },
                {
                    "buffer_command": {
                        "ispyb_command": "insert_particle_classification"
                    },
                    "buffer_lookup": {"particle_classification_group_id": 5},
                    "buffer_store": 11,
                    "class_distribution": "0.6",
                    "class_image_full_path": (
                        f"{tmp_path}/Class3D/job015/run_it020_class002.mrc"
                    ),
                    "class_number": 2,
                    "estimated_resolution": 10.0,
                    "ispyb_command": "buffer",
                    "overall_fourier_completeness": 0.9,
                    "particles_per_class": 60000.0,
                    "rotation_accuracy": "20.2",
                    "translation_accuracy": "22.2",
                    "angular_efficiency": 0.6,
                    "suggested_tilt": 30,
                },
            ],
        },
    )
    mock_recwrap_send.assert_any_call(
        "murfey_feedback",
        {
            "register": "done_3d_batch",
            "refine_dir": f"{tmp_path}/Refine3D/job",
            "class3d_dir": f"{tmp_path}/Class3D/job015",
            "best_class": 0,
            "do_refinement": False,
        },
    )

    assert mock_efficiency.call_count == 2


@mock.patch("cryoemservices.services.class3d.run_class3d")
def test_class3d_service_failed_resends(mock_class3d, offline_transport, tmp_path):
    """Failures of the processing should lead to reinjection of the message"""

    def raise_exception(*args, **kwargs):
        raise ValueError

    mock_class3d.side_effect = raise_exception

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    class3d_test_message = {
        "batch_size": "100000",
        "class_uuids": "{'0': 10, '1': 11}",
        "class3d_dir": f"{tmp_path}/Class3D/job015",
        "class3d_grp_uuid": "5",
        "class3d_nr_classes": "2",
        "do_initial_model": False,
        "initial_model_file": f"{tmp_path}/initial_model.mrc",
        "mask_diameter": "190.0",
        "particle_diameter": "180",
        "particles_file": f"{tmp_path}/Select/job013/particles_100000.star",
        "picker_id": "6",
        "relion_options": {},
    }
    end_message = copy.deepcopy(class3d_test_message)

    # Set up and run the service
    service = Class3D(environment={"queue": ""}, rabbitmq_credentials=Path("."))
    service._transport = offline_transport
    service.initializing()
    service.class3d(None, header=header, message=class3d_test_message)

    end_message["requeue"] = 1
    offline_transport.send.assert_any_call("class3d", end_message)
    offline_transport.ack.assert_called_once()


def test_class3d_service_nack_on_requeue(offline_transport, tmp_path):
    """Messages reinjected 5 times should nack"""
    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    class3d_test_message = {
        "batch_size": "100000",
        "class_uuids": "{'0': 10, '1': 11}",
        "class3d_dir": f"{tmp_path}/Class3D/job015",
        "class3d_grp_uuid": "5",
        "class3d_nr_classes": "2",
        "do_initial_model": False,
        "initial_model_file": f"{tmp_path}/initial_model.mrc",
        "mask_diameter": "190.0",
        "particle_diameter": "180",
        "particles_file": f"{tmp_path}/Select/job013/particles_100000.star",
        "picker_id": "6",
        "relion_options": {},
        "requeue": 5,
    }

    # Set up and run the service
    service = Class3D(environment={"queue": ""}, rabbitmq_credentials=Path("."))
    service._transport = offline_transport
    service.initializing()
    service.class3d(None, header=header, message=class3d_test_message)

    assert offline_transport.send.call_count == 0
    offline_transport.nack.assert_called_once()


def test_class3d_service_nack_wrong_params(offline_transport, tmp_path):
    """Messages without required parameters should nack"""
    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    class3d_test_message = {
        "relion_options": {},
    }

    # Set up and run the service
    service = Class3D(environment={"queue": ""}, rabbitmq_credentials=Path("."))
    service._transport = offline_transport
    service.initializing()
    service.class3d(None, header=header, message=class3d_test_message)

    assert offline_transport.send.call_count == 0
    offline_transport.nack.assert_called_once()
