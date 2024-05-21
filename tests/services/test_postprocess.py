from __future__ import annotations

import sys
from unittest import mock

import pytest
import zocalo.configuration
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import postprocess
from cryoemservices.util.spa_relion_service_options import RelionServiceOptions


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
@mock.patch("cryoemservices.services.postprocess.subprocess.run")
def test_postprocess_first_refine(
    mock_subprocess, mock_environment, offline_transport, tmp_path
):
    """
    Send a test message to the PostProcess service for a first Refinement job
    This should run particle selection and launch re-extraction jobs with slurm
    then send messages on to Murfey, ispyb and the node_creator
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = (
        "+ apply b-factor of: 50\n+ FINAL RESOLUTION: 4.5\n".encode("ascii")
    )
    mock_subprocess().stderr = "stderr".encode("ascii")

    # Create the expected input files
    (tmp_path / "Refine3D/job013").mkdir(parents=True)
    with open(tmp_path / "Refine3D/job013/run_model.star", "w") as f:
        f.write(
            "data_model_classes\n\nloop_\n_rlnReferenceImage\n"
            "_Angles\n_Rotation\n_Translation\n_Resolution\n_FourierCompleteness\n"
            "image 0 10 20 4 90"
        )

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    postprocess_test_message = {
        "parameters": {
            "half_map": str(tmp_path / "Refine3D/job013/half_map1.mrc"),
            "mask": str(tmp_path / "MaskCreate/job014/mask.mrc"),
            "rescaled_class_reference": str(tmp_path / "class_ref.mrc"),
            "job_dir": str(tmp_path / "PostProcess/job015"),
            "is_first_refinement": True,
            "pixel_size": 1.0,
            "number_of_particles": 5,
            "batch_size": 5,
            "class_number": 1,
            "postprocess_lowres": 10,
            "symmetry": "C1",
            "picker_id": 1,
            "refined_grp_uuid": 2,
            "refined_class_uuid": 3,
            "relion_options": {},
        },
        "content": "dummy",
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["batch_size"] = 5
    output_relion_options["pixel_size"] = 1.0
    output_relion_options.update(
        postprocess_test_message["parameters"]["relion_options"]
    )

    # Set up the mock service and call it
    service = postprocess.PostProcess(environment=mock_environment)
    service.transport = offline_transport
    service.start()
    service.postprocess(None, header=header, message=postprocess_test_message)

    postprocess_command = [
        "relion_postprocess",
        "--i",
        postprocess_test_message["parameters"]["half_map"],
        "--o",
        str(tmp_path / "PostProcess/job015/postprocess"),
        "--mask",
        postprocess_test_message["parameters"]["mask"],
        "--angpix",
        "1.0",
        "--auto_bfac",
        "--autob_lowres",
        "10.0",
        "--pipeline_control",
        f"{tmp_path}/PostProcess/job015/",
    ]

    assert mock_subprocess.call_count == 4
    mock_subprocess.assert_called_with(postprocess_command, capture_output=True)

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        destination="murfey_feedback",
        message={
            "parameters": {
                "register": "done_refinement",
                "project_dir": str(tmp_path),
                "resolution": 4.5,
                "number_of_particles": 5,
                "refined_grp_uuid": 2,
                "refined_class_uuid": 3,
                "class_reference": postprocess_test_message["parameters"][
                    "rescaled_class_reference"
                ],
                "class_number": 1,
                "mask_file": postprocess_test_message["parameters"]["mask"],
                "pixel_size": 1.0,
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="ispyb_connector",
        message={
            "parameters": {
                "ispyb_command": "multipart_message",
                "ispyb_command_list": [
                    {
                        "batch_number": "1",
                        "buffer_command": {
                            "ispyb_command": "insert_particle_classification_group"
                        },
                        "buffer_store": 2,
                        "ispyb_command": "buffer",
                        "number_of_classes_per_batch": "1",
                        "number_of_particles_per_batch": 5,
                        "particle_picker_id": 1,
                        "symmetry": "C1",
                        "type": "3D",
                    },
                    {
                        "buffer_command": {
                            "ispyb_command": "insert_particle_classification"
                        },
                        "buffer_lookup": {"particle_classification_group_id": 2},
                        "buffer_store": 3,
                        "class_distribution": 1,
                        "class_image_full_path": f"{tmp_path}/PostProcess/job015/postprocess.mrc",
                        "class_number": 1,
                        "estimated_resolution": 4.0,
                        "ispyb_command": "buffer",
                        "overall_fourier_completeness": 90.0,
                        "particles_per_class": 5,
                        "rotation_accuracy": "10",
                        "selected": "1",
                        "translation_accuracy": "20",
                    },
                    {
                        "buffer_command": {"ispyb_command": "insert_bfactor_fit"},
                        "buffer_lookup": {"particle_classification_id": 3},
                        "ispyb_command": "buffer",
                        "number_of_particles": 5,
                        "particle_batch_size": 5,
                        "resolution": 4.5,
                    },
                ],
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "job_type": "relion.postprocess",
                "input_file": (
                    postprocess_test_message["parameters"]["half_map"]
                    + ":"
                    + postprocess_test_message["parameters"]["mask"]
                ),
                "output_file": f"{tmp_path}/PostProcess/job015/postprocess.mrc",
                "relion_options": output_relion_options,
                "command": " ".join(postprocess_command),
                "stdout": "+ apply b-factor of: 50\n+ FINAL RESOLUTION: 4.5\n",
                "stderr": "stderr",
                "success": True,
            },
            "content": "dummy",
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.postprocess.subprocess.run")
def test_postprocess_bfactor(
    mock_subprocess, mock_environment, offline_transport, tmp_path
):
    """
    Send a test message to the PostProcess service for a B-Factor job
    This should run particle selection and launch re-extraction jobs with slurm
    then send messages on to Murfey, ispyb and the node_creator
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = (
        "+ apply b-factor of: 50\n+ FINAL RESOLUTION: 4.5\n".encode("ascii")
    )
    mock_subprocess().stderr = "stderr".encode("ascii")

    # Create the expected input files
    (tmp_path / "Refine3D/job013").mkdir(parents=True)
    with open(tmp_path / "Refine3D/job013/run_model.star", "w") as f:
        f.write(
            "data_model_classes\n\nloop_\n_rlnReferenceImage\n"
            "_Angles\n_Rotation\n_Translation\n_Resolution\n_FourierCompleteness\n"
            "image 0 10 20 4 90"
        )

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    postprocess_test_message = {
        "parameters": {
            "half_map": str(tmp_path / "Refine3D/job013/half_map1.mrc"),
            "mask": str(tmp_path / "MaskCreate/job014/mask.mrc"),
            "rescaled_class_reference": str(tmp_path / "class_ref.mrc"),
            "job_dir": str(tmp_path / "PostProcess/job015"),
            "is_first_refinement": False,
            "pixel_size": 1.0,
            "number_of_particles": 2,
            "batch_size": 5,
            "class_number": 1,
            "postprocess_lowres": 10,
            "symmetry": "C1",
            "picker_id": 1,
            "refined_grp_uuid": 2,
            "refined_class_uuid": 3,
            "relion_options": {},
        },
        "content": "dummy",
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["batch_size"] = 5
    output_relion_options["pixel_size"] = 1.0
    output_relion_options.update(
        postprocess_test_message["parameters"]["relion_options"]
    )

    # Set up the mock service and call it
    service = postprocess.PostProcess(environment=mock_environment)
    service.transport = offline_transport
    service.start()
    service.postprocess(None, header=header, message=postprocess_test_message)

    # Check the sends which differ from above
    # So no need to recheck postprocessing command or node creator message
    offline_transport.send.assert_any_call(
        destination="murfey_feedback",
        message={
            "parameters": {
                "register": "done_bfactor",
                "resolution": 4.5,
                "number_of_particles": 2,
                "refined_class_uuid": 3,
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="ispyb_connector",
        message={
            "parameters": {
                "ispyb_command": "multipart_message",
                "ispyb_command_list": [
                    {
                        "buffer_command": {"ispyb_command": "insert_bfactor_fit"},
                        "buffer_lookup": {"particle_classification_id": 3},
                        "ispyb_command": "buffer",
                        "number_of_particles": 2,
                        "particle_batch_size": 5,
                        "resolution": 4.5,
                    }
                ],
            },
            "content": "dummy",
        },
    )
