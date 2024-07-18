from __future__ import annotations

import sys
from unittest import mock

import pytest
import zocalo.configuration
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.util.spa_relion_service_options import RelionServiceOptions
from cryoemservices.wrappers import class3d_wrapper


@pytest.fixture
def mock_zocalo_configuration(tmp_path):
    mock_zc = mock.MagicMock(zocalo.configuration.Configuration)
    mock_zc.storage = {
        "zocalo.recipe_directory": tmp_path,
    }
    return mock_zc


@pytest.fixture
def mock_environment(mock_zocalo_configuration):
    return {"config": mock_zocalo_configuration}


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.wrappers.class3d_wrapper.subprocess.run")
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
def test_class3d_wrapper_do_initial_model(
    mock_recwrap_send, mock_subprocess, mock_environment, offline_transport, tmp_path
):
    """
    Send a test message to the Class3D wrapper for a first round of 50000 particles,
    without a provided initial model,
    which should then do images, ispyb and node_creator message sends,
    and tell murfey it has run
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("utf8")
    mock_subprocess().stderr = "stderr".encode("utf8")

    # Example recipe wrapper message to run the service with a few parameters varied
    class3d_test_message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "job_parameters": {
                    "allow_coarser": False,
                    "batch_size": "50000",
                    "class_uuids": "{'0': 10, '1': 11}",
                    "class3d_dir": f"{tmp_path}/Class3D/job015",
                    "class3d_grp_uuid": "5",
                    "class3d_nr_classes": "2",
                    "class3d_nr_iter": "25",
                    "ctf_intact_first_peak": False,
                    "do_ctf": True,
                    "do_norm": True,
                    "do_scale": True,
                    "do_zero_mask": True,
                    "do_initial_model": True,
                    "dont_combine_weights_via_disc": True,
                    "dont_correct_greyscale": True,
                    "fast_subsets": False,
                    "flattern_solvent": True,
                    "fn_mask": None,
                    "gpus": "0",
                    "healpix_order": 2,
                    "highres_limit": None,
                    "initial_lowpass": "20.0",
                    "initial_model_file": None,
                    "initial_model_gpus": "0",
                    "initial_model_iterations": 10,
                    "initial_model_offset_range": 6,
                    "initial_model_offset_step": 2,
                    "mask_diameter": "190.0",
                    "mpi_run_command": "srun -n 9",
                    "nr_pool": 5,
                    "offset_range": 5,
                    "offset_step": 2,
                    "oversampling": 1,
                    "pad": 2,
                    "particle_diameter": "180",
                    "particles_file": f"{tmp_path}/Select/job013/particles_50000.star",
                    "picker_id": "6",
                    "preread_images": False,
                    "relion_options": {},
                    "scratch_dir": None,
                    "skip_align": False,
                    "skip_gridding": False,
                    "start_initial_model_C1": True,
                    "symmetry": "C3",
                    "tau_fudge": 4,
                    "threads": 4,
                },
                "parameters": {
                    "cluster": {
                        "gpus": 4,
                        "tasks": 9,
                    },
                    "recipewrapper": f"{tmp_path}/Class3D/job015/.recipewrap",
                    "workingdir": f"{tmp_path}/Class3D/job015/",
                },
                "queue": "cluster.submission",
                "service": "Class3DWrapper",
                "wrapper": {"task_information": "Class3D"},
            },
        },
        "recipe-pointer": 1,
        "environment": {"ID": "envID"},
        "recipe-path": [],
        "payload": [],
    }
    output_relion_options = RelionServiceOptions()
    output_relion_options.particle_diameter = 180
    output_relion_options.class3d_nr_classes = 2
    output_relion_options.batch_size = 50
    output_relion_options = dict(output_relion_options)

    # Create the expected output files
    (tmp_path / "InitialModel/job014").mkdir(parents=True, exist_ok=True)
    (tmp_path / "InitialModel/job014/initial_model.mrc").touch()
    with open(tmp_path / "InitialModel/job014/run_it010_model.star", "w") as ini_star:
        ini_star.write(
            "data_model_classes\nloop_\n"
            "_rlnReferenceImage\n_rlnClassDistribution\n_rlnEstimatedResolution\n"
            "1@InitialModel/job014/run_it010_classes.mrcs 0.4 30.3\n"
        )

    (tmp_path / "Class3D/job015").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "Class3D/job015/run_it025_data.star", "w") as data_star:
        data_star.write(
            "data_particles\nloop_\n_rlnAngleRot\n_rlnAngleTilt\n_rlnClassNumber\n"
            "0.5 1.0 1\n1.5 2.0 1\n2.5 3.0 2\n3.5 4.0 2\n"
        )
    with open(tmp_path / "Class3D/job015/run_it025_model.star", "w") as model_star:
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

    # Set up and run the mock service
    service_wrapper = class3d_wrapper.Class3DWrapper(environment=mock_environment)
    service_wrapper.set_recipe_wrapper(recipe_wrapper)
    service_wrapper.run()

    # Check the initial model command
    initial_model_command = ["relion_initial_model"]
    mock_subprocess.assert_called_with(
        initial_model_command, capture_output=True, cwd=str(tmp_path)
    )
    align_symmetry_command = ["relion_align_symmetry"]
    mock_subprocess.assert_called_with(
        align_symmetry_command, capture_output=True, cwd=str(tmp_path)
    )
    # Check the node creator and murfey sends for the initial model
    mock_recwrap_send.assert_any_call(
        "murfey_feedback",
        {
            "job_type": "relion.initialmodel",
            "input_file": f"{tmp_path}//particles_50000.star",
            "output_file": f"{tmp_path}/InitialModel/job015/initial_model.mrc",
            "relion_options": output_relion_options,
            "command": " ".join(initial_model_command),
            "stdout": "stdout",
            "stderr": "stderr",
            "success": True,
        },
    )
    mock_recwrap_send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.initialmodel",
            "input_file": f"{tmp_path}/Select/job013/particles_50000.star",
            "output_file": f"{tmp_path}/InitialModel/job015/initial_model.mrc",
            "relion_options": output_relion_options,
            "command": " ".join(align_symmetry_command),
            "stdout": "stdout",
            "stderr": "stderr",
            "success": True,
        },
    )
    mock_recwrap_send.assert_any_call(
        "murfey_feedback",
        {
            "register": "save_initial_model",
            "job_dir": f"{tmp_path}/InitialModel/job014/initial_model.mrc",
        },
    )

    # Check the expected command was run
    class2d_command = [
        "srun",
        "-n",
        "9",
        "relion_refine_mpi",
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
        "--iter",
        "25",
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
            "file": f"{tmp_path}/Class2D/job010/run_it025_classes.mrcs",
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
                        f"{tmp_path}/Class2D/job010/run_it025_classes_1.jpeg"
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
                        f"{tmp_path}/Class2D/job010/run_it025_classes_2.jpeg"
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
