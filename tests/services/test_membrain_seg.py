from __future__ import annotations

import json
import os
import sys
from unittest import mock

import pytest
import zocalo.configuration
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import membrain_seg


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
@mock.patch("cryoemservices.util.slurm_submission.subprocess.run")
def test_membrain_seg_service(
    mock_subprocess,
    mock_environment,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to membrain-seg for the slurm submission version
    This should call the mock subprocess then send messages to the images service.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = (
        '{"job_id": "1", "jobs": [{"job_state": ["COMPLETED"]}]}'.encode("ascii")
    )
    mock_subprocess().stderr = "stderr".encode("ascii")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    segmentation_test_message = {
        "parameters": {
            "tomogram": f"{tmp_path}/Denoise/job007/tomograms/test_stack_aretomo.denoised.mrc",
            "output_dir": f"{tmp_path}/Segmentation/job008/tomograms",
            "model_checkpoint": "checkpoint.ckpt",
            "pixel_size": "1.0",
            "rescale_patches": True,
            "augmentation": True,
            "store_probabilities": True,
            "store_connected_components": True,
            "window_size": 100,
            "connected_component_threshold": 2,
            "segmentation_threshold": 4,
            "cleanup_output": False,
        },
        "content": "dummy",
    }

    # Set up the mock service
    service = membrain_seg.MembrainSeg(environment=mock_environment)
    service.transport = offline_transport
    service.start()

    # Construct the file which contains rest api submission information
    os.environ["DENOISING_SIF"] = "topaz.sif"
    os.environ["SLURM_RESTAPI_CONFIG"] = str(tmp_path / "restapi.txt")
    with open(tmp_path / "restapi.txt", "w") as restapi_config:
        restapi_config.write(
            "user: user\n"
            "user_home: /home\n"
            f"user_token: {tmp_path}/token.txt\n"
            "required_directories: [directory1, directory2]\n"
            "partition: partition\n"
            "partition_preference: preference\n"
            "cluster: cluster\n"
            "url: /url/of/slurm/restapi\n"
            "api_version: v0.0.40\n"
        )
    with open(tmp_path / "token.txt", "w") as token:
        token.write("token_key")

    # Touch the expected output files
    (tmp_path / "Segmentation/job008/tomograms").mkdir(parents=True)
    (
        tmp_path
        / "Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc.out"
    ).touch()
    (
        tmp_path
        / "Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc.err"
    ).touch()

    # Send a message to the service
    service.membrain_seg(None, header=header, message=segmentation_test_message)

    # Check the slurm commands were run
    slurm_submit_command = (
        f'curl -H "X-SLURM-USER-NAME:user" -H "X-SLURM-USER-TOKEN:token_key" '
        '-H "Content-Type: application/json" -X POST '
        "/url/of/slurm/restapi/slurm/v0.0.40/job/submit "
        f"-d @{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc.json"
    )
    slurm_status_command = (
        'curl -H "X-SLURM-USER-NAME:user" -H "X-SLURM-USER-TOKEN:token_key" '
        '-H "Content-Type: application/json" -X GET '
        "/url/of/slurm/restapi/slurm/v0.0.40/job/1"
    )
    mock_subprocess.assert_any_call(
        slurm_submit_command, capture_output=True, shell=True
    )
    mock_subprocess.assert_any_call(
        slurm_status_command, capture_output=True, shell=True
    )
    assert mock_subprocess.call_count == 5

    # Check the images service request
    assert offline_transport.send.call_count == 3
    offline_transport.send.assert_any_call(
        destination="images",
        message={
            "image_command": "mrc_central_slice",
            "file": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "skip_rescaling": True,
        },
    )
    offline_transport.send.assert_any_call(
        destination="movie",
        message={
            "image_command": "mrc_to_apng",
            "file": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "skip_rescaling": True,
        },
    )
    offline_transport.send.assert_any_call(
        destination="ispyb_connector",
        message={
            "parameters": {
                "ispyb_command": "insert_processed_tomogram",
                "file_path": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
                "processing_type": "Segmented",
            },
            "content": {"dummy": "dummy"},
        },
    )

    # Check the segmentation command
    with open(
        tmp_path
        / "Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc.json",
        "r",
    ) as script_file:
        script_json = json.load(script_file)
    segmentation_command = script_json["script"].split("\n")[-1]

    expected_command = [
        "membrain",
        "segment",
        "--out-folder",
        f"{tmp_path}/Segmentation/job008/tomograms",
        "--tomogram-path",
        f"{tmp_path}/Denoise/job007/tomograms/test_stack_aretomo.denoised.mrc",
        "--ckpt-path",
        "checkpoint.ckpt",
        "--in-pixel-size",
        "1.0",
        "--sliding-window-size",
        "100",
        "--connected-component-thres",
        "2",
        "--segmentation-threshold",
        "4.0",
        "--rescale-patches",
        "--test-time-augmentation",
        "--store-probabilities",
        "--store-connected-components",
    ]
    assert segmentation_command == " ".join(expected_command)
