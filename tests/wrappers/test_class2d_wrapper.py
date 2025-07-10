from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import pytest
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.util.relion_service_options import RelionServiceOptions
from cryoemservices.wrappers import class2d_wrapper


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.wrappers.class2d_wrapper.subprocess.run")
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
def test_class2d_wrapper_incomplete_batch(
    mock_recwrap_send, mock_subprocess, offline_transport, tmp_path
):
    """
    Send a test message to the Class2D wrapper for an incomplete batch,
    which should then do images, ispyb and node_creator message sends,
    and tell murfey it has run
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("utf8")
    mock_subprocess().stderr = "stderr".encode("utf8")

    # Example recipe wrapper message to run the service with a few parameters varied
    class2d_test_message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "job_parameters": {
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
                "parameters": {
                    "cluster": {
                        "gpus": 4,
                        "tasks": 9,
                    },
                    "recipewrapper": f"{tmp_path}/Class2D/job010/.recipewrap",
                    "workingdir": f"{tmp_path}/Class2D/job010/",
                },
                "queue": "cluster.submission",
                "service": "Class2DWrapper",
                "wrapper": {"task_information": "Class2D"},
            },
        },
        "recipe-pointer": 1,
        "environment": {"ID": "envID"},
        "recipe-path": [],
        "payload": [],
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

    # Set up and run the mock service
    service_wrapper = class2d_wrapper.Class2DWrapper(recipe_wrapper)
    service_wrapper.run()

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


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.wrappers.class2d_wrapper.subprocess.run")
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
def test_class2d_wrapper_complete_batch(
    mock_recwrap_send, mock_subprocess, offline_transport, tmp_path
):
    """
    Send a test message to the Class2D wrapper for a complete batch,
    which should then do images, ispyb and node_creator message sends,
    and also start icebreaker and class selection jobs
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("utf8")
    mock_subprocess().stderr = "stderr".encode("utf8")

    # Example recipe wrapper message to run the service with
    class2d_test_message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "job_parameters": {
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
                },
            },
        },
        "recipe-pointer": 1,
        "environment": {"ID": "envID"},
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

    # Create a recipe wrapper with the test message
    recipe_wrapper = RecipeWrapper(
        message=class2d_test_message, transport=offline_transport
    )

    # Set up and run the mock service
    service_wrapper = class2d_wrapper.Class2DWrapper(recipe_wrapper)
    service_wrapper.run()

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
    assert mock_recwrap_send.call_count == 5
    # Don't need to re-test the node_creator, images or ispyb messages
    mock_recwrap_send.assert_any_call(
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
    mock_recwrap_send.assert_any_call(
        "select_classes",
        {
            "input_file": f"{tmp_path}/Class2D/job010/run_it020_optimiser.star",
            "relion_options": output_relion_options,
            "class_uuids": "{'0': 10}",
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.wrappers.class2d_wrapper.subprocess.run")
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
def test_class2d_wrapper_cryodann(
    mock_recwrap_send, mock_subprocess, offline_transport, tmp_path
):
    """
    Send a test message to the Class2D wrapper for a complete batch with cryodann on
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("utf8")
    mock_subprocess().stderr = "stderr".encode("utf8")

    # Example recipe wrapper message to run the service with
    class2d_test_message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "job_parameters": {
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
                    "do_cryodann": True,
                    "cryodann_dataset": "cryodann/dataset",
                    "relion_options": {},
                },
            },
        },
        "recipe-pointer": 1,
        "environment": {"ID": "envID"},
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

    # Create cryodann outputs
    (tmp_path / "Class2D/job010/cryodann/lightning_logs").mkdir(parents=True)
    np.save(
        tmp_path / "Class2D/job010/cryodann/lightning_logs/scores.npy",
        np.array([1, 2, 3, 4, 5]),
    )

    # Create a recipe wrapper with the test message
    recipe_wrapper = RecipeWrapper(
        message=class2d_test_message, transport=offline_transport
    )

    # Set up and run the mock service
    service_wrapper = class2d_wrapper.Class2DWrapper(recipe_wrapper)
    service_wrapper.run()

    # Check the expected command was run
    assert mock_subprocess.call_count == 9
    particle_alignment_command = [
        "relion_stack_create",
        "--i",
        f"{tmp_path}/Class2D/job010/run_it020_data.star",
        "--o",
        f"{tmp_path}/Class2D/job010/aligned_particles/aligned",
    ]
    lowpass_command = [
        "relion_image_handler",
        "--i",
        f"{tmp_path}/Class2D/job010/aligned_particles/aligned.mrcs",
        "--lowpass",
        "10",
        "--o",
        f"{tmp_path}/Class2D/job010/aligned_particles/aligned_lowpassed.mrcs",
    ]
    cryovae_command = [
        "cryovae",
        f"{tmp_path}/Class2D/job010/aligned_particles/aligned_lowpassed.mrcs",
        f"{tmp_path}/Class2D/job010/cryodann/cryovae",
        "--beta=0.1",
    ]
    cryodann_command = [
        "cryodann",
        "cryodann/dataset",
        f"{tmp_path}/Class2D/job010/cryodann/cryovae/recons.mrcs",
        f"{tmp_path}/Class2D/job010/cryodann",
        "--particle_file",
        f"{tmp_path}/Class2D/job010/aligned_particles/aligned.star",
        "--keep_percent",
        "0.5",
    ]

    mock_subprocess.assert_any_call(
        particle_alignment_command, cwd=str(tmp_path), capture_output=True
    )
    mock_subprocess.assert_any_call(
        lowpass_command, cwd=str(tmp_path), capture_output=True
    )
    mock_subprocess.assert_any_call(
        cryovae_command, cwd=str(tmp_path), capture_output=True
    )
    mock_subprocess.assert_any_call(
        cryodann_command, cwd=str(tmp_path), capture_output=True
    )

    # Check cryodann wrote scores
    with open(tmp_path / "Class2D/job010/run_it020_data.star", "r") as data_star:
        output_particles_data = data_star.read()
    assert output_particles_data == (
        "data_optics\nloop_\n_rlnImagePixelSize\n2.5\n\n"
        "data_particles\nloop_\n_rlnCoordinateX\n_rlnCryodannScore\n"
        "1 1\n2 2\n3 3\n4 4\n5 5\n"
    )

    # Check the expected message sends to other processes
    assert mock_recwrap_send.call_count == 5
    # Don't need to re-test the node_creator, images or ispyb messages
    mock_recwrap_send.assert_any_call(
        "select_classes",
        {
            "input_file": f"{tmp_path}/Class2D/job010/run_it020_optimiser.star",
            "relion_options": output_relion_options,
            "class_uuids": "{'0': 10}",
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.wrappers.class2d_wrapper.subprocess.run")
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
def test_class2d_wrapper_rerun_buffer_lookup(
    mock_recwrap_send, mock_subprocess, offline_transport, tmp_path
):
    """
    Send a test message to the Class2D wrapper for a re-run incomplete batch.
    This should cause buffer lookups in the ispyb message
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("utf8")
    mock_subprocess().stderr = "stderr".encode("utf8")

    # Example recipe wrapper message to run the service with
    class2d_test_message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "job_parameters": {
                    "batch_is_complete": False,
                    "batch_size": "50000",
                    "class2d_dir": f"{tmp_path}/Class2D/job010",
                    "class2d_grp_uuid": "5",
                    "class2d_nr_classes": "1",
                    "class_uuids": "{'0': 10}",
                    "do_icebreaker_jobs": True,
                    "do_vdam": False,
                    "gpus": "0:1:2:3",
                    "particle_diameter": "180",
                    "particles_file": f"{tmp_path}/Select/job009/particles_split1.star",
                    "picker_id": "6",
                    "relion_options": {},
                },
            },
        },
        "recipe-pointer": 1,
        "environment": {"ID": "envID"},
    }

    # Create the expected output files
    (tmp_path / "Class2D/job010").mkdir(parents=True, exist_ok=True)
    (tmp_path / "Class2D/job010/RELION_JOB_EXIT_SUCCESS").touch()
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

    # Create a recipe wrapper with the test message
    recipe_wrapper = RecipeWrapper(
        message=class2d_test_message, transport=offline_transport
    )

    # Set up and run the mock service
    service_wrapper = class2d_wrapper.Class2DWrapper(recipe_wrapper)
    service_wrapper.run()

    # Check the expected message send to ispyb
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
                    "buffer_lookup": {"particle_classification_group_id": 5},
                    "ispyb_command": "buffer",
                    "number_of_classes_per_batch": 1,
                    "number_of_particles_per_batch": 5,
                    "particle_picker_id": 6,
                    "symmetry": "C1",
                    "type": "2D",
                },
                {
                    "buffer_command": {
                        "ispyb_command": "insert_particle_classification"
                    },
                    "buffer_lookup": {
                        "particle_classification_group_id": 5,
                        "particle_classification_id": 10,
                    },
                    "class_distribution": "0.4",
                    "class_image_full_path": (
                        f"{tmp_path}/Class2D/job010/run_it020_classes_1.jpeg"
                    ),
                    "class_number": 1,
                    "estimated_resolution": 12.2,
                    "ispyb_command": "buffer",
                    "overall_fourier_completeness": 1.0,
                    "particles_per_class": 2.0,
                    "rotation_accuracy": "30.3",
                    "translation_accuracy": "33.3",
                },
            ],
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.wrappers.class2d_wrapper.subprocess.run")
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
def test_class2d_wrapper_failure_releases_hold(
    mock_recwrap_send, mock_subprocess, offline_transport, tmp_path
):
    """
    Send a test message to the Class2D wrapper which will fail the job.
    This should send a release message to murfey and report failure to the node creator
    """
    mock_subprocess().returncode = 1
    mock_subprocess().stdout = "stdout".encode("utf8")
    mock_subprocess().stderr = "stderr".encode("utf8")

    # Example recipe wrapper message to run the service with
    class2d_test_message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "job_parameters": {
                    "batch_is_complete": False,
                    "batch_size": "50000",
                    "class2d_dir": f"{tmp_path}/Class2D/job010",
                    "class2d_grp_uuid": "5",
                    "class2d_nr_classes": "1",
                    "class_uuids": "{'0': 10}",
                    "do_icebreaker_jobs": True,
                    "do_vdam": False,
                    "gpus": "0:1:2:3",
                    "particle_diameter": "180",
                    "particles_file": f"{tmp_path}/Select/job009/particles_split1.star",
                    "picker_id": "6",
                    "relion_options": {},
                },
            },
        },
        "recipe-pointer": 1,
        "environment": {"ID": "envID"},
    }

    # Create the expected output files
    (tmp_path / "Class2D/job010").mkdir(parents=True, exist_ok=True)
    (tmp_path / "Class2D/job010/RELION_JOB_EXIT_SUCCESS").touch()
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

    # Create a recipe wrapper with the test message
    recipe_wrapper = RecipeWrapper(
        message=class2d_test_message, transport=offline_transport
    )

    # Set up and run the mock service
    service_wrapper = class2d_wrapper.Class2DWrapper(recipe_wrapper)
    service_wrapper.run()

    # Check the expected message sends to the node creator and murfey
    assert mock_recwrap_send.call_count == 2
    mock_recwrap_send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.class2d.em",
            "input_file": f"{tmp_path}/Select/job009/particles_split1.star",
            "output_file": f"{tmp_path}/Class2D/job010",
            "relion_options": mock.ANY,
            "command": mock.ANY,
            "stdout": "stdout",
            "stderr": "stderr",
            "success": False,
        },
    )
    mock_recwrap_send.assert_any_call(
        "murfey_feedback",
        {
            "register": "done_incomplete_2d_batch",
            "job_dir": f"{tmp_path}/Class2D/job010",
        },
    )
