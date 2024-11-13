from __future__ import annotations

import json
import os
import sys
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import denoise_slurm
from cryoemservices.util.relion_service_options import RelionServiceOptions


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.util.slurm_submission.subprocess.run")
def test_denoise_slurm_service(
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to Denoising for the slurm submission version
    This should call the mock subprocess then send messages on to
    the membrain-seg and images services.
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
    denoise_test_message = {
        "parameters": {
            "volume": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
            "output_dir": f"{tmp_path}/Denoise/job007/denoised",
            "suffix": ".denoised",
            "model": "unet-3d",
            "even_train_path": None,
            "odd_train_path": None,
            "n_train": 1000,
            "n_test": 200,
            "crop": 96,
            "base_kernel_width": 11,
            "optim": "adagrad",
            "lr": "0.001",
            "criteria": "L2",
            "momentum": "0.8",
            "batch_size": 10,
            "num_epochs": 500,
            "weight_decay": 0,
            "save_interval": 10,
            "save_prefix": "prefix",
            "num_workers": 1,
            "num_threads": 0,
            "gaussian": 0,
            "patch_size": 96,
            "patch_padding": 48,
            "device": "-2",
            "cleanup_output": False,
            "relion_options": {},
        },
        "content": "dummy",
    }
    output_relion_options = dict(RelionServiceOptions())

    # Set up the mock service
    service = denoise_slurm.DenoiseSlurm()
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
    (tmp_path / "Denoise/job007/denoised").mkdir(parents=True)
    (tmp_path / "Denoise/job007/denoised/test_stack_aretomo.denoised.mrc.out").touch()
    (tmp_path / "Denoise/job007/denoised/test_stack_aretomo.denoised.mrc.err").touch()

    # Send a message to the service
    service.denoise(None, header=header, message=denoise_test_message)

    # Check the slurm commands were run
    slurm_submit_command = (
        f'curl -H "X-SLURM-USER-NAME:user" -H "X-SLURM-USER-TOKEN:token_key" '
        '-H "Content-Type: application/json" -X POST '
        "/url/of/slurm/restapi/slurm/v0.0.40/job/submit "
        f"-d @{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc.json"
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

    # Check file transfer and retrieval
    # assert mock_transfer.call_count == 1
    # mock_transfer.assert_any_call([f"{tmp_path}/test_stack_aretomo.mrc"])
    # assert mock_retrieve.call_count == 1
    # mock_retrieve.assert_any_call(
    #    job_directory=tmp_path,
    #    files_to_skip=[tmp_path / "test_stack_aretomo.mrc"],
    #    basepath="test_stack_aretomo",
    # )
    assert mock_subprocess.call_count == 5

    # Check the denoising command
    with open(
        tmp_path / "Denoise/job007/denoised/test_stack_aretomo.denoised.mrc.json", "r"
    ) as script_file:
        script_json = json.load(script_file)
    topaz_command = script_json["script"].split("\n")[-1]

    denoise_command = [
        "topaz",
        "denoise3d",
        f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
        "-o",
        f"{tmp_path}/Denoise/job007/denoised",
        "--suffix",
        ".denoised",
        "-m",
        "unet-3d",
        "--N-train",
        "1000",
        "--N-test",
        "200",
        "-c",
        "96",
        "--base-kernel-width",
        "11",
        "--optim adagrad",
        "--lr",
        "0.001",
        "--criteria",
        "L2",
        "--momentum",
        "0.8",
        "--batch-size",
        "10",
        "--num-epochs",
        "500",
        "-w",
        "0",
        "--save-interval",
        "10",
        "--save-prefix",
        "prefix",
        "--num-workers",
        "1",
        "-j",
        "0",
        "-g",
        "0",
        "-s",
        "96",
        "-p",
        "48",
        "-d",
        "-2",
    ]
    assert topaz_command == " ".join(denoise_command)

    # Check the images service request
    assert offline_transport.send.call_count == 5
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "experiment_type": "tomography",
                "job_type": "relion.denoisetomo",
                "input_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
                "output_file": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
                "relion_options": output_relion_options,
                "command": " ".join(denoise_command),
                "stdout": "",
                "stderr": "",
                "success": True,
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="images",
        message={
            "image_command": "mrc_central_slice",
            "file": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
        },
    )
    offline_transport.send.assert_any_call(
        destination="movie",
        message={
            "image_command": "mrc_to_apng",
            "file": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
        },
    )
    offline_transport.send.assert_any_call(
        destination="ispyb_connector",
        message={
            "parameters": {
                "ispyb_command": "insert_processed_tomogram",
                "file_path": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
                "processing_type": "Denoised",
            },
            "content": {"dummy": "dummy"},
        },
    )
    offline_transport.send.assert_any_call(
        destination="segmentation",
        message={
            "tomogram": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
            "output_dir": f"{tmp_path}/Segmentation/job008/tomograms",
        },
    )
