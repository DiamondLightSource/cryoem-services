from __future__ import annotations

from unittest import mock

import pytest
from requests import Response
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import membrain_seg
from cryoemservices.util.relion_service_options import RelionServiceOptions
from tests.test_utils.config import cluster_submission_configuration


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@mock.patch("cryoemservices.services.membrain_seg.segment")
def test_membrain_seg_service_local_memseg(
    mock_segment,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to membrain-seg for the version running without a subprocess
    This should call the mock subprocess then send messages to the images service.
    """

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    segmentation_test_message = {
        "tomogram": f"{tmp_path}/Denoise/job007/tomograms/test_stack_aretomo.denoised.mrc",
        "output_dir": f"{tmp_path}/Segmentation/job008/tomograms",
        "pretrained_checkpoint": "checkpoint.ckpt",
        "pixel_size": "1.0",
        "rescale_patches": True,
        "augmentation": True,
        "store_probabilities": True,
        "store_connected_components": True,
        "window_size": 100,
        "connected_component_threshold": 2,
        "segmentation_threshold": 4,
        "relion_options": {},
    }
    output_relion_options = dict(RelionServiceOptions())

    # Set up the mock service and send a message to it
    service = membrain_seg.MembrainSeg(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.membrain_seg(None, header=header, message=segmentation_test_message)

    # Check the membrain command was run
    membrain_command = [
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
    mock_segment.assert_called_once_with(
        tomogram_path=f"{tmp_path}/Denoise/job007/tomograms/test_stack_aretomo.denoised.mrc",
        ckpt_path="checkpoint.ckpt",
        out_folder=f"{tmp_path}/Segmentation/job008/tomograms",
        rescale_patches=True,
        in_pixel_size=1.0,
        out_pixel_size=10.0,
        store_probabilities=True,
        sw_roi_size=100,
        store_connected_components=True,
        connected_component_thres=2,
        test_time_augmentation=True,
        segmentation_threshold=4.0,
    )

    # Check the images service request
    assert offline_transport.send.call_count == 4
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "experiment_type": "tomography",
            "job_type": "membrain.segment",
            "input_file": f"{tmp_path}/Denoise/job007/tomograms/test_stack_aretomo.denoised.mrc",
            "output_file": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "relion_options": output_relion_options,
            "command": " ".join(membrain_command),
            "stdout": "",
            "stderr": "",
            "success": True,
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_central_slice",
            "file": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "skip_rescaling": True,
        },
    )
    offline_transport.send.assert_any_call(
        "movie",
        {
            "image_command": "mrc_to_apng",
            "file": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "skip_rescaling": True,
        },
    )
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "ispyb_command": "insert_processed_tomogram",
            "file_path": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "processing_type": "Segmented",
        },
    )


@mock.patch("cryoemservices.services.membrain_seg.run_subprocess")
@mock.patch("cryoemservices.services.membrain_seg.subprocess.run")
def test_membrain_seg_service_local_subprocess(
    mock_subprocess,
    mock_skip_imports,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to membrain-seg for the local version
    This should call the mock subprocess then send messages to the images service.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    segmentation_test_message = {
        "tomogram": f"{tmp_path}/Denoise/job007/tomograms/test_stack_aretomo.denoised.mrc",
        "output_dir": f"{tmp_path}/Segmentation/job008/tomograms",
        "pretrained_checkpoint": "checkpoint.ckpt",
        "pixel_size": "1.0",
        "rescale_patches": True,
        "augmentation": True,
        "store_probabilities": True,
        "store_connected_components": True,
        "window_size": 100,
        "connected_component_threshold": 2,
        "segmentation_threshold": 4,
        "relion_options": {},
    }
    output_relion_options = dict(RelionServiceOptions())

    # Set up the mock service and send a message to it
    service = membrain_seg.MembrainSeg(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.membrain_seg(None, header=header, message=segmentation_test_message)

    # Check the membrain command was run
    membrain_command = [
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
    assert mock_subprocess.call_count == 4
    mock_subprocess.assert_any_call(membrain_command, capture_output=True)

    # Check the images service request
    assert offline_transport.send.call_count == 4
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "experiment_type": "tomography",
            "job_type": "membrain.segment",
            "input_file": f"{tmp_path}/Denoise/job007/tomograms/test_stack_aretomo.denoised.mrc",
            "output_file": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "relion_options": output_relion_options,
            "command": " ".join(membrain_command),
            "stdout": "stdout",
            "stderr": "stderr",
            "success": True,
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_central_slice",
            "file": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "skip_rescaling": True,
        },
    )
    offline_transport.send.assert_any_call(
        "movie",
        {
            "image_command": "mrc_to_apng",
            "file": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "skip_rescaling": True,
        },
    )
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "ispyb_command": "insert_processed_tomogram",
            "file_path": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "processing_type": "Segmented",
        },
    )


@mock.patch("cryoemservices.util.slurm_submission.requests")
def test_membrain_seg_service_slurm(
    mock_requests,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to membrain-seg for the slurm submission version
    This should call the mock subprocess then send messages to the images service.
    """
    # Set up the returned job number
    response_object = Response()
    response_object._content = (
        '{"job_id": 1, "error_code": 0, "error": "", "jobs": [{"job_state": ["COMPLETED"]}]}'
    ).encode("utf8")
    response_object.status_code = 200
    mock_requests.Session().post.return_value = response_object
    mock_requests.Session().get.return_value = response_object

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    segmentation_test_message = {
        "tomogram": f"{tmp_path}/Denoise/job007/tomograms/test_stack_aretomo.denoised.mrc",
        "output_dir": f"{tmp_path}/Segmentation/job008/tomograms",
        "pretrained_checkpoint": "checkpoint.ckpt",
        "pixel_size": "1.0",
        "rescale_patches": True,
        "augmentation": True,
        "store_probabilities": True,
        "store_connected_components": True,
        "window_size": 100,
        "connected_component_threshold": 2,
        "segmentation_threshold": 4,
        "cleanup_output": False,
        "submit_to_slurm": True,
        "relion_options": {},
    }
    output_relion_options = dict(RelionServiceOptions())

    # Construct the file which contains rest api submission information
    cluster_submission_configuration(tmp_path)

    # Set up the mock service
    service = membrain_seg.MembrainSeg(
        environment={
            "config": f"{tmp_path}/config.yaml",
            "slurm_cluster": "default",
            "queue": "",
        },
        transport=offline_transport,
    )
    service.initializing()

    # Touch the expected output files
    (tmp_path / "Segmentation/job008/tomograms").mkdir(parents=True)
    (
        tmp_path
        / "Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.out"
    ).touch()
    (
        tmp_path
        / "Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.err"
    ).touch()

    # Send a message to the service
    service.membrain_seg(None, header=header, message=segmentation_test_message)

    # Check the segmentation command
    segment_command = [
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
                "source /etc/profile.d/modules.sh\n"
                "module load EM/membrain-seg\n" + " ".join(segment_command)
            ),
            "job": {
                "cpus_per_task": 1,
                "current_working_directory": f"{tmp_path}/Segmentation/job008/tomograms",
                "standard_output": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.out",
                "standard_error": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.err",
                "environment": ["USER=user", "HOME=/home"],
                "name": "membrain-seg",
                "nodes": "1",
                "partition": "partition",
                "prefer": "preference",
                "tasks": 1,
                "memory_per_node": {"number": 25000, "set": True, "infinite": False},
                "time_limit": {"number": 60, "set": True, "infinite": False},
                "tres_per_job": "gres/gpu:1",
            },
        },
    )
    mock_requests.Session().get.assert_called_with(
        url="/url/of/slurm/restapi/slurm/v0.0.40/job/1"
    )

    # Check the images service request
    assert offline_transport.send.call_count == 4
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "experiment_type": "tomography",
            "job_type": "membrain.segment",
            "input_file": f"{tmp_path}/Denoise/job007/tomograms/test_stack_aretomo.denoised.mrc",
            "output_file": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "relion_options": output_relion_options,
            "command": " ".join(segment_command),
            "stdout": "",
            "stderr": "",
            "success": True,
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_central_slice",
            "file": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "skip_rescaling": True,
        },
    )
    offline_transport.send.assert_any_call(
        "movie",
        {
            "image_command": "mrc_to_apng",
            "file": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "skip_rescaling": True,
        },
    )
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "ispyb_command": "insert_processed_tomogram",
            "file_path": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "processing_type": "Segmented",
        },
    )


@mock.patch("cryoemservices.services.membrain_seg.segment")
def test_membrain_seg_service_local_memseg_rerun(
    mock_segment,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to membrain-seg for the version running without a subprocess
    This should call the mock subprocess then send messages to the images service.
    This does a rerun so should not request node creation
    """

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    segmentation_test_message = {
        "tomogram": f"{tmp_path}/Denoise/job007/tomograms/test_stack_aretomo.denoised.mrc",
        "output_dir": f"{tmp_path}/Segmentation/job008/tomograms",
        "pretrained_checkpoint": "checkpoint.ckpt",
        "pixel_size": "1.0",
        "rescale_patches": True,
        "augmentation": True,
        "store_probabilities": True,
        "store_connected_components": True,
        "window_size": 100,
        "connected_component_threshold": 2,
        "segmentation_threshold": 4,
        "relion_options": {},
    }

    # Pre-make the output so this is a rerun
    (tmp_path / "Segmentation/job008/tomograms").mkdir(parents=True)
    (
        tmp_path
        / "Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc"
    ).touch()

    # Set up the mock service and send a message to it
    service = membrain_seg.MembrainSeg(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.membrain_seg(None, header=header, message=segmentation_test_message)

    # Check the images service request
    assert offline_transport.send.call_count == 3
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_central_slice",
            "file": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "skip_rescaling": True,
        },
    )
    offline_transport.send.assert_any_call(
        "movie",
        {
            "image_command": "mrc_to_apng",
            "file": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "skip_rescaling": True,
        },
    )
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "ispyb_command": "insert_processed_tomogram",
            "file_path": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo.denoised_segmented.mrc",
            "processing_type": "Segmented",
        },
    )
