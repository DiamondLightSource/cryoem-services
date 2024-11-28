from __future__ import annotations

import sys
from unittest import mock

import pytest
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.util.relion_service_options import RelionServiceOptions
from cryoemservices.wrappers import class3d_wrapper


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.wrappers.class3d_wrapper.subprocess.run")
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
def test_class3d_wrapper_do_initial_model(
    mock_recwrap_send, mock_subprocess, offline_transport, tmp_path
):
    """
    Send a test message to the Class3D wrapper for a first round of 50000 particles,
    without a provided initial model.
    The initial model and 3D classification commands should be run,
    then both cause ispyb, node_creator and murfey messages.
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
    output_relion_options.batch_size = 50000
    output_relion_options.initial_lowpass = 20
    output_relion_options.symmetry = "C3"
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
    service_wrapper = class3d_wrapper.Class3DWrapper()
    service_wrapper.set_recipe_wrapper(recipe_wrapper)
    service_wrapper.run()

    # Check the initial model command
    assert mock_subprocess.call_count == 6
    initial_model_command = [
        "relion_refine",
        "--grad",
        "--denovo_3dref",
        "--i",
        "Select/job013/particles_50000.star",
        "--o",
        "InitialModel/job014/run",
        "--particle_diameter",
        str(output_relion_options["mask_diameter"]),
        "--gpu",
        "0",
        "--sym",
        "C1",
        "--iter",
        "10",
        "--offset_range",
        "6.0",
        "--offset_step",
        "2.0",
        "--dont_combine_weights_via_disc",
        "--pool",
        "5",
        "--pad",
        "2",
        "--ctf",
        "--K",
        "2",
        "--flatten_solvent",
        "--zero_mask",
        "--oversampling",
        "1",
        "--healpix_order",
        "2",
        "--j",
        "4",
        "--pipeline_control",
        "InitialModel/job014/",
    ]
    mock_subprocess.assert_any_call(
        initial_model_command, capture_output=True, cwd=str(tmp_path)
    )
    align_symmetry_command = [
        "relion_align_symmetry",
        "--i",
        "InitialModel/job014/run_it010_model.star",
        "--o",
        "InitialModel/job014/initial_model.mrc",
        "--sym",
        str(output_relion_options["symmetry"]),
        "--apply_sym",
        "--select_largest_class",
        "--pipeline_control",
        "InitialModel/job014/",
    ]
    mock_subprocess.assert_any_call(
        align_symmetry_command, capture_output=True, cwd=str(tmp_path)
    )
    # Check the node creator and murfey sends for the initial model
    mock_recwrap_send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.initialmodel",
            "input_file": f"{tmp_path}/Select/job013/particles_50000.star",
            "output_file": f"{tmp_path}/InitialModel/job014/initial_model.mrc",
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
            "output_file": f"{tmp_path}/InitialModel/job014/initial_model.mrc",
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
            "initial_model": f"{tmp_path}/InitialModel/job014/initial_model.mrc",
        },
    )

    # Check the expected 3D classifcation command was run
    class3d_command = [
        "srun",
        "-n",
        "9",
        "relion_refine_mpi",
        "--i",
        "Select/job013/particles_50000.star",
        "--o",
        "Class3D/job015/run",
        "--ref",
        f"{tmp_path}/InitialModel/job014/initial_model.mrc",
        "--particle_diameter",
        str(output_relion_options["mask_diameter"]),
        "--dont_combine_weights_via_disc",
        "--pool",
        "5",
        "--pad",
        "2",
        "--firstiter_cc",
        "--ini_high",
        "20.0",
        "--ctf",
        "--iter",
        "25",
        "--tau2_fudge",
        "4.0",
        "--K",
        "2",
        "--flatten_solvent",
        "--zero_mask",
        "--oversampling",
        "1",
        "--healpix_order",
        "2",
        "--offset_range",
        "5.0",
        "--offset_step",
        "2.0",
        "--sym",
        str(output_relion_options["symmetry"]),
        "--norm",
        "--scale",
        "--j",
        "4",
        "--gpu",
        "0",
        "--pipeline_control",
        "Class3D/job015/",
    ]
    mock_subprocess.assert_any_call(
        class3d_command, capture_output=True, cwd=str(tmp_path)
    )

    # Check the expected message sends to other processes
    assert mock_recwrap_send.call_count == 6
    mock_recwrap_send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.class3d",
            "input_file": f"{tmp_path}/Select/job013/particles_50000.star:{tmp_path}/InitialModel/job014/initial_model.mrc",
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
                    "buffer_command": {
                        "ispyb_command": "insert_particle_classification_group"
                    },
                    "buffer_store": 5,
                    "ispyb_command": "buffer",
                    "number_of_classes_per_batch": 2,
                    "number_of_particles_per_batch": 50000,
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
                        f"{tmp_path}/Class3D/job015/run_it025_class001.mrc"
                    ),
                    "class_number": 1,
                    "estimated_resolution": 12.2,
                    "ispyb_command": "buffer",
                    "overall_fourier_completeness": 1.0,
                    "particles_per_class": 20000.0,
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
                        f"{tmp_path}/Class3D/job015/run_it025_class002.mrc"
                    ),
                    "class_number": 2,
                    "estimated_resolution": 10.0,
                    "ispyb_command": "buffer",
                    "overall_fourier_completeness": 0.9,
                    "particles_per_class": 30000.0,
                    "rotation_accuracy": "20.2",
                    "translation_accuracy": "22.2",
                },
                {
                    "buffer_command": {"ispyb_command": "insert_cryoem_initial_model"},
                    "buffer_lookup": {"particle_classification_id": 10},
                    "ispyb_command": "buffer",
                    "number_of_particles": 20000.0,
                    "resolution": "30.3",
                    "store_result": "ispyb_initial_model_id",
                },
                {
                    "buffer_command": {"ispyb_command": "insert_cryoem_initial_model"},
                    "buffer_lookup": {"particle_classification_id": 11},
                    "cryoem_initial_model_id": "$ispyb_initial_model_id",
                    "ispyb_command": "buffer",
                    "number_of_particles": 20000.0,
                    "resolution": "30.3",
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


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.wrappers.class3d_wrapper.subprocess.run")
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
def test_class3d_wrapper_has_initial_model(
    mock_recwrap_send, mock_subprocess, offline_transport, tmp_path
):
    """
    Send a test message to the Class3D wrapper for a second round of 100000 particles,
    with a provided initial model.
    The 3D classification command should be run,
    then cause ispyb, node_creator and murfey messages.
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
    output_relion_options.batch_size = 100000
    output_relion_options.class3d_nr_iter = 20
    output_relion_options = dict(output_relion_options)

    # Create the expected output files
    (tmp_path / "initial_model.mrc").touch()

    (tmp_path / "Class3D/job015").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "Class3D/job015/run_it020_data.star", "w") as data_star:
        data_star.write(
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

    # Set up and run the mock service
    service_wrapper = class3d_wrapper.Class3DWrapper()
    service_wrapper.set_recipe_wrapper(recipe_wrapper)
    service_wrapper.run()

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


best_class_test_matrix = (
    # tuple of Fractions, Resolutions, Completenesses, Do refine?, Best class
    ([0.1, 0.2, 0.3, 0.4], [8, 9, 10, 11], [0.95, 0.95, 0.95, 0.95], True, 1),
    # ^ Pick best resolution
    ([0.1, 0.2, 0.3, 0.4], [8, 9, 10, 11], [0.8, 0.95, 0.95, 0.95], True, 2),
    # ^ Pick second best resolution due to completeness
    ([0.1, 0.4, 0.3, 0.2], [8, 8, 8, 8], [0.95, 0.95, 0.95, 0.95], True, 2),
    # ^ Pick highest particle count at best resolution
    ([0.1, 0.2, 0.3, 0.4], [11, 12, 13, 14], [0.95, 0.95, 0.95, 0.95], False, 0),
    # ^ Don't refine, bad resolution
    ([0.1, 0.2, 0.3, 0.4], [8, 9, 8, 11], [0.9, 0.8, 0.7, 0.6], False, 0),
    # ^ Don't refine, bad completeness
)


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@pytest.mark.parametrize("test_classes", best_class_test_matrix)
@mock.patch("cryoemservices.wrappers.class3d_wrapper.subprocess.run")
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
def test_class3d_wrapper_for_refinement(
    mock_recwrap_send,
    mock_subprocess,
    test_classes: tuple[list[float], list[float], list[float], bool, int],
    offline_transport,
    tmp_path,
):
    """
    Send a test message to the Class3D wrapper for a final round of 200000 particles,
    with a provided initial model.
    The 3D classification commands should be run,
    then cause ispyb, node_creator and murfey messages, requesting refinement.
    Runs a variety of different cases to test the estimation of the best class
    """
    mock_subprocess().returncode = 0

    # Example recipe wrapper message to run the service with a few parameters varied
    class3d_test_message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "job_parameters": {
                    "batch_size": "200000",
                    "class_uuids": "{'0': 10, '1': 11, '2': 12, '3': 13}",
                    "class3d_dir": f"{tmp_path}/Class3D/job015",
                    "class3d_grp_uuid": "5",
                    "initial_model_file": f"{tmp_path}/initial_model.mrc",
                    "mask_diameter": "190.0",
                    "particle_diameter": "180",
                    "particles_file": f"{tmp_path}/Select/job013/particles_200000.star",
                    "picker_id": "6",
                    "relion_options": {},
                },
                "parameters": {
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
    }

    # Create the expected output files
    (tmp_path / "initial_model.mrc").touch()

    (tmp_path / "Class3D/job015").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "Class3D/job015/run_it020_data.star", "w") as data_star:
        data_star.write(
            "data_particles\nloop_\n_rlnAngleRot\n_rlnAngleTilt\n_rlnClassNumber\n"
            "0.5 1.0 1\n1.5 2.0 1\n2.5 3.0 2\n3.5 4.0 2\n"
        )
    with open(tmp_path / "Class3D/job015/run_it020_model.star", "w") as model_star:
        model_star.write(
            "data_model_classes\nloop_\n"
            "_rlnReferenceImage\n_Fraction\n_Rotation\n_Translation\n"
            "_Resolution\n_Completeness\n_OffsetX\n_OffsetY\n"
            f"1@Class3D {test_classes[0][0]} 0 0 "
            f"{test_classes[1][0]} {test_classes[2][0]} 0 0\n"
            f"2@Class3D {test_classes[0][1]} 0 0 "
            f"{test_classes[1][1]} {test_classes[2][1]} 0 0\n"
            f"3@Class3D {test_classes[0][2]} 0 0 "
            f"{test_classes[1][2]} {test_classes[2][2]} 0 0\n"
            f"4@Class3D {test_classes[0][3]} 0 0 "
            f"{test_classes[1][3]} {test_classes[2][3]} 0 0\n"
        )

    # Create a recipe wrapper with the test message
    recipe_wrapper = RecipeWrapper(
        message=class3d_test_message, transport=offline_transport
    )

    # Set up and run the mock service
    service_wrapper = class3d_wrapper.Class3DWrapper()
    service_wrapper.set_recipe_wrapper(recipe_wrapper)
    service_wrapper.run()

    # No need to check classification command and ispyb messages again
    assert mock_recwrap_send.call_count == 3
    # Check the expected message sends to murfey
    mock_recwrap_send.assert_any_call(
        "murfey_feedback",
        {
            "register": "done_3d_batch",
            "refine_dir": f"{tmp_path}/Refine3D/job",
            "class3d_dir": f"{tmp_path}/Class3D/job015",
            "best_class": test_classes[4],
            "do_refinement": test_classes[3],
        },
    )
