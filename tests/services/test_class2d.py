from __future__ import annotations

import copy
from unittest import mock

import pytest
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.class2d import Class2D
from cryoemservices.util.relion_service_options import RelionServiceOptions


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "ack")
    mocker.spy(transport, "nack")
    return transport


@mock.patch("cryoemservices.wrappers.class2d_wrapper.subprocess.run")
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
def test_class2d_service_incomplete_batch(
    mock_recwrap_send, mock_subprocess, offline_transport, tmp_path
):
    """
    Send a test message to the Class2D service for an incomplete batch,
    which should then do images, ispyb and node_creator message sends,
    and tell murfey it has run
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("utf8")
    mock_subprocess().stderr = "stderr".encode("utf8")

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    class2d_test_message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "parameters": {
                    "allow_coarser": False,
                    "batch_is_complete": False,
                    "batch_size": "50",
                    "centre_classes": True,
                    "class2d_dir": f"{tmp_path}/Class2D/job010",
                    "class2d_grp_uuid": "5",
                    "class2d_nr_classes": "2",
                    "class2d_nr_iter": "25",
                    "class_uuids": "{'0': 10, '1': 11}",
                    "ctf_intact_first_peak": False,
                    "do_ctf": True,
                    "do_icebreaker_jobs": True,
                    "do_norm": True,
                    "do_scale": True,
                    "do_vdam": True,
                    "do_zero_mask": True,
                    "dont_combine_weights_via_disc": True,
                    "flattern_solvent": True,
                    "gpus": "0",
                    "highres_limit": None,
                    "mask_diameter": "190.0",
                    "mpi_run_command": "srun -n 9",
                    "nr_pool": 5,
                    "offset_range": 5,
                    "offset_step": 2,
                    "oversampling": 1,
                    "pad": 2,
                    "particle_diameter": "180",
                    "particles_file": f"{tmp_path}/Select/job009/particles_split1.star",
                    "picker_id": "6",
                    "preread_images": False,
                    "psi_step": 6.0,
                    "relion_options": {},
                    "scratch_dir": None,
                    "skip_align": False,
                    "skip_gridding": False,
                    "tau_fudge": 4,
                    "threads": 4,
                    "vdam_threshold": 0.1,
                    "vdam_write_iter": 10,
                    "vdam_mini_batches": 200,
                    "vdam_subset": 7000,
                    "vdam_initial_fraction": 0.3,
                    "vdam_final_fraction": 0.1,
                },
            },
        },
        "recipe-pointer": 1,
    }
    output_relion_options = RelionServiceOptions()
    output_relion_options.particle_diameter = 180
    output_relion_options.class2d_nr_classes = 2
    output_relion_options.batch_size = 50
    output_relion_options = dict(output_relion_options)

    # Create the expected output files
    (tmp_path / "Class2D/job010").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "Class2D/job010/run_it200_data.star", "w") as data_star:
        data_star.write(
            "data_optics\nloop_\n_rlnImagePixelSize\n2.5\n\n"
            "data_particles\nloop_\n_rlnCoordinateX\n1\n2\n3\n4\n5"
        )
    with open(tmp_path / "Class2D/job010/run_it200_model.star", "w") as model_star:
        model_star.write(
            "data_model_classes\nloop_\n"
            "_rlnReferenceImage\n_vdam1\n_vdam2\n_Fraction\n_Rotation\n_Translation\n"
            "_Resolution\n_Completeness\n_OffsetX\n_OffsetY\n"
            "1@Class2D/job010/run_it020_classes.mrcs vdam vdam 0.4 30.3 33.3 12.2 1.0 0.6 0.01\n"
            "2@Class2D/job010/run_it020_classes.mrcs vdam vdam 0.6 20.2 22.2 10.0 0.9 -0.5 -0.02"
        )

    # Create a recipe wrapper with the test message
    recipe_wrapper = RecipeWrapper(
        message=class2d_test_message, transport=offline_transport
    )

    # Set up and run the service
    service = Class2D(environment={"queue": ""}, transport=offline_transport)
    service.initializing()
    service.class2d(rw=recipe_wrapper, header=header, message=None)

    # Check the expected command was run
    class2d_command = [
        "relion_refine",
        "--i",
        "Select/job009/particles_split1.star",
        "--o",
        "Class2D/job010/run",
        "--particle_diameter",
        str(output_relion_options["mask_diameter"]),
        "--dont_combine_weights_via_disc",
        "--pool",
        "5",
        "--pad",
        "2",
        "--ctf",
        "--tau2_fudge",
        "4.0",
        "--K",
        "2",
        "--flatten_solvent",
        "--zero_mask",
        "--center_classes",
        "--oversampling",
        "1",
        "--psi_step",
        "6.0",
        "--offset_range",
        "5.0",
        "--offset_step",
        "2.0",
        "--norm",
        "--scale",
        "--j",
        "4",
        "--gpu",
        "0",
        "--pipeline_control",
        "Class2D/job010/",
        "--grad",
        "--class_inactivity_threshold",
        "0.1",
        "--grad_write_iter",
        "10",
        "--grad_fin_subset",
        "7000",
        "--grad_ini_frac",
        "0.3",
        "--grad_fin_frac",
        "0.1",
        "--iter",
        "200",
    ]
    mock_subprocess.assert_called_with(
        class2d_command, capture_output=True, cwd=str(tmp_path)
    )

    # Check the expected message sends to other processes
    assert mock_recwrap_send.call_count == 4
    mock_recwrap_send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.class2d.vdam",
            "input_file": f"{tmp_path}/Select/job009/particles_split1.star",
            "output_file": f"{tmp_path}/Class2D/job010",
            "relion_options": output_relion_options,
            "command": " ".join(class2d_command),
            "stdout": "stdout",
            "stderr": "stderr",
            "success": True,
        },
    )
    mock_recwrap_send.assert_any_call(
        "images",
        {
            "image_command": "mrc_to_jpeg",
            "file": f"{tmp_path}/Class2D/job010/run_it200_classes.mrcs",
            "all_frames": "True",
        },
    )
    mock_recwrap_send.assert_any_call(
        "ispyb_connector",
        {
            "ispyb_command": "multipart_message",
            "ispyb_command_list": [
                {
                    "batch_number": 1,
                    "binned_pixel_size": "2.5",
                    "buffer_command": {
                        "ispyb_command": "insert_particle_classification_group"
                    },
                    "buffer_store": 5,
                    "ispyb_command": "buffer",
                    "number_of_classes_per_batch": 2,
                    "number_of_particles_per_batch": 5,
                    "particle_picker_id": 6,
                    "symmetry": "C1",
                    "type": "2D",
                },
                {
                    "buffer_command": {
                        "ispyb_command": "insert_particle_classification"
                    },
                    "buffer_lookup": {"particle_classification_group_id": 5},
                    "buffer_store": 10,
                    "class_distribution": "0.4",
                    "class_image_full_path": (
                        f"{tmp_path}/Class2D/job010/run_it200_classes_1.jpeg"
                    ),
                    "class_number": 1,
                    "estimated_resolution": 12.2,
                    "ispyb_command": "buffer",
                    "overall_fourier_completeness": 1.0,
                    "particles_per_class": 2.0,
                    "rotation_accuracy": "30.3",
                    "translation_accuracy": "33.3",
                },
                {
                    "buffer_command": {
                        "ispyb_command": "insert_particle_classification"
                    },
                    "buffer_lookup": {"particle_classification_group_id": 5},
                    "buffer_store": 11,
                    "class_distribution": "0.6",
                    "class_image_full_path": (
                        f"{tmp_path}/Class2D/job010/run_it200_classes_2.jpeg"
                    ),
                    "class_number": 2,
                    "estimated_resolution": 10.0,
                    "ispyb_command": "buffer",
                    "overall_fourier_completeness": 0.9,
                    "particles_per_class": 3.0,
                    "rotation_accuracy": "20.2",
                    "translation_accuracy": "22.2",
                },
            ],
        },
    )
    mock_recwrap_send.assert_any_call(
        "murfey_feedback",
        {
            "register": "done_incomplete_2d_batch",
            "job_dir": f"{tmp_path}/Class2D/job010",
        },
    )


@mock.patch("cryoemservices.wrappers.class2d_wrapper.subprocess.run")
def test_class2d_service_complete_batch(mock_subprocess, offline_transport, tmp_path):
    """
    Send a test message to the Class2D service for a complete batch,
    which should then do images, ispyb and node_creator message sends,
    and also start icebreaker and class selection jobs
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("utf8")
    mock_subprocess().stderr = "stderr".encode("utf8")

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    class2d_test_message = {
        "batch_is_complete": True,
        "batch_size": "50000",
        "class2d_dir": f"{tmp_path}/Class2D/job010",
        "class2d_grp_uuid": "5",
        "class2d_nr_classes": "1",
        "class_uuids": "{'0': 10}",
        "do_icebreaker_jobs": True,
        "do_vdam": False,
        "gpus": "0:1:2:3",
        "particle_diameter": "180",
        "particles_file": f"{tmp_path}/Select/job009/particles_split2.star",
        "picker_id": "6",
        "relion_options": {},
    }
    output_relion_options = RelionServiceOptions()
    output_relion_options.particle_diameter = 180
    output_relion_options.class2d_nr_classes = 1
    output_relion_options.class2d_nr_iter = 20
    output_relion_options = dict(output_relion_options)

    # Create the expected output files
    (tmp_path / "Class2D/job010").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "Class2D/job010/run_it020_data.star", "w") as data_star:
        data_star.write(
            "data_optics\nloop_\n_rlnImagePixelSize\n2.5\n\n"
            "data_particles\nloop_\n_rlnCoordinateX\n1\n2\n3\n4\n5"
        )
    with open(tmp_path / "Class2D/job010/run_it020_model.star", "w") as model_star:
        model_star.write(
            "data_model_classes\nloop_\n"
            "_rlnReferenceImage\n_Fraction\n_Rotation\n_Translation\n"
            "_Resolution\n_Completeness\n_OffsetX\n_OffsetY\n"
            "1@Class2D/job010/run_it020_classes.mrcs 0.4 30.3 33.3 12.2 1.0 0.6 0.01\n"
        )

    # Set up and run the service
    service = Class2D(environment={"queue": ""}, transport=offline_transport)
    service.initializing()
    service.class2d(None, header=header, message=class2d_test_message)

    # Check the expected command was run
    class2d_command = [
        "srun",
        "-n",
        "5",
        "relion_refine_mpi",
        "--i",
        "Select/job009/particles_split2.star",
        "--o",
        "Class2D/job010/run",
        "--particle_diameter",
        str(output_relion_options["mask_diameter"]),
        "--dont_combine_weights_via_disc",
        "--preread_images",
        "--pool",
        "100",
        "--pad",
        "2",
        "--ctf",
        "--tau2_fudge",
        "2",
        "--K",
        "1",
        "--flatten_solvent",
        "--zero_mask",
        "--center_classes",
        "--oversampling",
        "1",
        "--psi_step",
        "12.0",
        "--offset_range",
        "5",
        "--offset_step",
        "2",
        "--norm",
        "--scale",
        "--j",
        "8",
        "--gpu",
        "0:1:2:3",
        "--pipeline_control",
        "Class2D/job010/",
        "--iter",
        "20",
    ]
    mock_subprocess.assert_called_with(
        class2d_command, capture_output=True, cwd=str(tmp_path)
    )

    # Check the expected message sends to other processes
    assert offline_transport.send.call_count == 5
    # Don't need to re-test the node_creator, images or ispyb messages
    offline_transport.send.assert_any_call(
        "icebreaker",
        {
            "icebreaker_type": "particles",
            "input_micrographs": (
                f"{tmp_path}/IceBreaker/job003/grouped_micrographs.star"
            ),
            "input_particles": f"{tmp_path}/Select/job009/particles_split2.star",
            "output_path": f"{tmp_path}/IceBreaker/job011/",
            "mc_uuid": -1,
            "relion_options": output_relion_options,
        },
    )
    offline_transport.send.assert_any_call(
        "select_classes",
        {
            "input_file": f"{tmp_path}/Class2D/job010/run_it020_optimiser.star",
            "relion_options": output_relion_options,
            "class_uuids": "{'0': 10}",
        },
    )
    offline_transport.ack.assert_called_once()


@mock.patch("cryoemservices.services.class2d.run_class2d")
def test_class2d_service_failed_resends(mock_class2d, offline_transport, tmp_path):
    """Failures of the processing should lead to reinjection of the message"""

    def raise_exception(*args, **kwargs):
        raise ValueError

    mock_class2d.side_effect = raise_exception

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    class2d_test_message = {
        "batch_is_complete": True,
        "batch_size": "50000",
        "class2d_dir": f"{tmp_path}/Class2D/job010",
        "class2d_grp_uuid": "5",
        "class2d_nr_classes": "1",
        "class_uuids": "{'0': 10}",
        "gpus": "0:1:2:3",
        "particle_diameter": "180",
        "particles_file": f"{tmp_path}/Select/job009/particles_split2.star",
        "picker_id": "6",
        "relion_options": {},
    }
    end_message = copy.deepcopy(class2d_test_message)

    # Set up and run the service
    service = Class2D(environment={"queue": ""}, transport=offline_transport)
    service.initializing()
    service.class2d(None, header=header, message=class2d_test_message)

    end_message["requeue"] = 1
    offline_transport.send.assert_any_call("class2d", end_message)
    offline_transport.ack.assert_called_once()


def test_class2d_service_nack_on_requeue(offline_transport, tmp_path):
    """Messages reinjected 5 times should nack"""
    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    class2d_test_message = {
        "batch_is_complete": True,
        "batch_size": "50000",
        "class2d_dir": f"{tmp_path}/Class2D/job010",
        "class2d_grp_uuid": "5",
        "class2d_nr_classes": "1",
        "class_uuids": "{'0': 10}",
        "gpus": "0:1:2:3",
        "particle_diameter": "180",
        "particles_file": f"{tmp_path}/Select/job009/particles_split2.star",
        "picker_id": "6",
        "relion_options": {},
        "requeue": 5,
    }

    # Set up and run the service
    service = Class2D(environment={"queue": ""}, transport=offline_transport)
    service.initializing()
    service.class2d(None, header=header, message=class2d_test_message)

    assert offline_transport.send.call_count == 0
    offline_transport.nack.assert_called_once()


def test_class2d_service_nack_wrong_params(offline_transport, tmp_path):
    """Messages without required parameters should nack"""
    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    class2d_test_message = {
        "relion_options": {},
    }

    # Set up and run the service
    service = Class2D(environment={"queue": ""}, transport=offline_transport)
    service.initializing()
    service.class2d(None, header=header, message=class2d_test_message)

    assert offline_transport.send.call_count == 0
    offline_transport.nack.assert_called_once()
