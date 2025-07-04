from __future__ import annotations

import os
import sys
from unittest import mock

import pytest
from requests import Response
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import denoise, denoise_slurm
from cryoemservices.util.relion_service_options import RelionServiceOptions
from tests.test_utils.config import cluster_submission_configuration


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.denoise.subprocess.run")
def test_denoise_local_service(
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to Denoising for the locally-running version
    This should call the mock subprocess then send messages on to
    the membrain-seg and images services.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    denoise_test_message = {
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
        "relion_options": {"pixel_size_downscaled": 1},
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["pixel_size_downscaled"] = 1

    # Set up the mock service and send a message to the service
    service = denoise.Denoise(environment={"queue": ""}, transport=offline_transport)
    service.initializing()
    service.denoise(None, header=header, message=denoise_test_message)

    # Check the denoising command
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
        "--optim",
        "adagrad",
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
    mock_subprocess.assert_any_call(denoise_command, capture_output=True)

    # Check the images service request
    assert offline_transport.send.call_count == 6
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "experiment_type": "tomography",
            "job_type": "relion.denoisetomo",
            "input_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
            "output_file": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
            "relion_options": output_relion_options,
            "command": " ".join(denoise_command),
            "stdout": "stdout",
            "stderr": "stderr",
            "success": True,
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_central_slice",
            "file": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
        },
    )
    offline_transport.send.assert_any_call(
        "movie",
        {
            "image_command": "mrc_to_apng",
            "file": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
        },
    )
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "ispyb_command": "insert_processed_tomogram",
            "file_path": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
            "processing_type": "Denoised",
        },
    )
    offline_transport.send.assert_any_call(
        "segmentation",
        {
            "tomogram": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
            "output_dir": f"{tmp_path}/Segmentation/job008/tomograms",
            "pixel_size": "1.0",
            "relion_options": output_relion_options,
        },
    )
    offline_transport.send.assert_any_call(
        "cryolo",
        {
            "input_path": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
            "output_path": f"{tmp_path}/AutoPick/job009/CBOX_3D/test_stack_aretomo.denoised.cbox",
            "experiment_type": "tomography",
            "cryolo_box_size": 40,
            "relion_options": output_relion_options,
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.util.slurm_submission.requests")
@mock.patch("cryoemservices.services.denoise_slurm.transfer_files")
@mock.patch("cryoemservices.services.denoise_slurm.retrieve_files")
def test_denoise_slurm_service(
    mock_retrieve,
    mock_transfer,
    mock_requests,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to Denoising for the slurm submission version
    This should call the mock subprocess then send messages on to
    the membrain-seg and images services.
    """
    # Set up the returned job number
    response_object = Response()
    response_object._content = (
        '{"job_id": 1, "error_code": 0, "error": "", "jobs": [{"job_state": ["COMPLETED"]}]}'
    ).encode("utf8")
    response_object.status_code = 200
    mock_requests.Session().post.return_value = response_object
    mock_requests.Session().get.return_value = response_object

    mock_transfer.return_value = ["test_stack_aretomo.mrc"]

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    denoise_test_message = {
        "volume": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
        "output_dir": f"{tmp_path}/Denoise/job007/denoised",
        "cleanup_output": False,
        "relion_options": {"pixel_size_downscaled": 1},
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["pixel_size_downscaled"] = 1

    # Construct the file which contains rest api submission information
    os.environ["DENOISING_SIF"] = "topaz.sif"
    cluster_submission_configuration(tmp_path)

    # Set up the mock service
    service = denoise_slurm.DenoiseSlurm(
        environment={
            "config": f"{tmp_path}/config.yaml",
            "slurm_cluster": "default",
            "queue": "",
        },
        transport=offline_transport,
    )
    service.initializing()

    # Touch the expected output files
    (tmp_path / "Denoise/job007/denoised").mkdir(parents=True)
    (tmp_path / "Denoise/job007/denoised/test_stack_aretomo.denoised.mrc").touch()
    (tmp_path / "Denoise/job007/denoised/test_stack_aretomo.denoised.out").touch()
    (tmp_path / "Denoise/job007/denoised/test_stack_aretomo.denoised.err").touch()

    # Send a message to the service
    service.denoise(None, header=header, message=denoise_test_message)

    # Check the denoising command
    singularity_command = [
        "singularity",
        "exec",
        "--nv",
        "--bind",
        "/tmp/tmp_$SLURM_JOB_ID:/tmp,directory1,directory2",
        "--home",
        "/home",
        "topaz.sif",
    ]
    denoise_command = [
        "topaz",
        "denoise3d",
        f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
        "-o",
        f"{tmp_path}/Denoise/job007/denoised",
        "--suffix",
        ".denoised",
    ]
    singularity_command.extend(denoise_command)

    # Check the slurm commands were run
    mock_requests.Session.assert_called()
    mock_requests.Session().headers.__setitem__.assert_any_call(
        "X-SLURM-USER-NAME", "user"
    )
    mock_requests.Session().headers.__setitem__.assert_any_call(
        "X-SLURM-USER-TOKEN", "token_key"
    )
    mock_requests.Session().post.assert_called_with(
        url="/url/of/slurm/restapi/slurm/v0.0.40/job/submit",
        json={
            "script": (
                "#!/bin/bash\n"
                "echo \"$(date '+%Y-%m-%d %H:%M:%S.%3N'): running slurm job\"\n"
                "mkdir /tmp/tmp_$SLURM_JOB_ID\n"
                "export APPTAINER_CACHEDIR=/tmp/tmp_$SLURM_JOB_ID\n"
                "export APPTAINER_TMPDIR=/tmp/tmp_$SLURM_JOB_ID\n"
                "\n" + " ".join(singularity_command) + "\n"
                "rm -rf /tmp/tmp_$SLURM_JOB_ID"
            ),
            "job": {
                "cpus_per_task": 1,
                "current_working_directory": f"{tmp_path}/Denoise/job007/denoised",
                "standard_output": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.out",
                "standard_error": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.err",
                "environment": ["USER=user", "HOME=/home"],
                "name": "Denoising",
                "nodes": "1",
                "partition": "partition",
                "prefer": "preference",
                "tasks": 1,
                "memory_per_node": {"number": 12000, "set": True, "infinite": False},
                "time_limit": {"number": 60, "set": True, "infinite": False},
                "tres_per_job": "gres/gpu:1",
            },
        },
    )
    mock_requests.Session().get.assert_called_with(
        url="/url/of/slurm/restapi/slurm/v0.0.40/job/1"
    )

    # Check file transfer and retrieval
    assert mock_transfer.call_count == 1
    mock_transfer.assert_any_call(
        [tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc"]
    )
    assert mock_retrieve.call_count == 1
    mock_retrieve.assert_any_call(
        job_directory=tmp_path / "Denoise/job007/denoised",
        files_to_skip=[tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc"],
        basepath="test_stack_aretomo",
    )

    # Check the images service request
    assert offline_transport.send.call_count == 6
    offline_transport.send.assert_any_call(
        "node_creator",
        {
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
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_central_slice",
            "file": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
        },
    )
    offline_transport.send.assert_any_call(
        "movie",
        {
            "image_command": "mrc_to_apng",
            "file": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
        },
    )
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "ispyb_command": "insert_processed_tomogram",
            "file_path": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
            "processing_type": "Denoised",
        },
    )
    offline_transport.send.assert_any_call(
        "segmentation",
        {
            "tomogram": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
            "output_dir": f"{tmp_path}/Segmentation/job008/tomograms",
            "pixel_size": "1.0",
            "relion_options": output_relion_options,
        },
    )
    offline_transport.send.assert_any_call(
        "cryolo",
        {
            "input_path": f"{tmp_path}/Denoise/job007/denoised/test_stack_aretomo.denoised.mrc",
            "output_path": f"{tmp_path}/AutoPick/job009/CBOX_3D/test_stack_aretomo.denoised.cbox",
            "experiment_type": "tomography",
            "cryolo_box_size": 40,
            "relion_options": output_relion_options,
        },
    )
