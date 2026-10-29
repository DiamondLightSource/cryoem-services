from pathlib import Path
from unittest import mock

import pytest
from requests import Response
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import hole_segmentation
from tests.test_utils.config import cluster_submission_configuration


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "ack")
    mocker.spy(transport, "nack")
    return transport


@mock.patch("cryoemservices.services.hole_segmentation.segment_single_image")
def test_hole_segmentation_local(mock_segment, offline_transport, tmp_path):
    """
    Send a test message to HoleSegmentation for local version
    """
    mock_segment.return_value = (
        {"beam_edge_detected": False, "hole_detected": False, "hole_percent": 50.5},
        {},
    )

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    segment_test_message = {
        "micrograph": "/path/to/micrograph.mrc",
        "output_dir": "/path/to/output",
        "checkpoint": "/path/to/checkpoint",
        "submit_to_slurm": False,
        "points_per_side": 8,
        "binning": 4,
        "min_area_fraction": 0.001,
        "beam_threshold": 10,
        "hole_threshold": 100,
        "save_summary": False,
    }

    # Set up the mock service
    service = hole_segmentation.HoleSegmentation(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()

    # Set some parameters then send a message to the service
    service.defocus1 = 1
    service.defocus2 = 2
    service.astigmatism_angle = 3
    service.cc_value = 4
    service.estimated_resolution = 5
    service.segment_hole(None, header=header, message=segment_test_message)

    # Check segmentation call
    mock_segment.assert_called_once_with(
        impath=Path("/path/to/micrograph.mrc"),
        checkpoint=Path("/path/to/checkpoint"),
        output_dir=Path("/path/to/output"),
        points_per_side=8,
        binning=4,
        min_area_fraction=0.001,
        beam_threshold=10,
        hole_threshold=100,
        save_summary=False,
    )

    # Check that the correct messages were sent
    offline_transport.send.assert_called_once_with(
        "murfey_feedback",
        {
            "register": "segmented_hole",
            "micrograph": "/path/to/micrograph.mrc",
            "hole_percent": 50.5,
            "hole_detected": False,
            "beam_edge_detected": False,
        },
    )


@mock.patch("cryoemservices.util.slurm_submission.requests")
def test_hole_segmentation_slurm(mock_requests, offline_transport, tmp_path):
    """
    Send a test message to the hole segmentation to go to slurm
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
    segment_test_message = {
        "micrograph": "/path/to/micrograph.mrc",
        "output_dir": f"{tmp_path}/output",
        "checkpoint": "/path/to/checkpoint",
        "submit_to_slurm": True,
        "save_summary": True,
    }

    # Construct the file which contains rest api submission information
    cluster_submission_configuration(tmp_path)

    # Touch the expected output files
    (tmp_path / "output").mkdir()
    with open(tmp_path / "output/micrograph_mask.out", "w") as f:
        f.write("Hole area = 95.5%\nBeam edge detected\n")
    (tmp_path / "output/micrograph_mask.err").touch()

    # Set up the mock service and call it
    service = hole_segmentation.HoleSegmentation(
        environment={
            "config": f"{tmp_path}/config.yaml",
            "slurm_cluster": "default",
            "queue": "",
        },
        transport=offline_transport,
    )
    service.initializing()
    service.segment_hole(None, header=header, message=segment_test_message)

    # Get the expected commands
    segment_command = [
        "cryoemservices.segment_holes",
        "--micrograph_dir",
        str("/path/to"),
        "--input_glob",
        "micrograph.mrc",
        "--output_dir",
        f"{tmp_path}/output",
        "--checkpoint",
        "/path/to/checkpoint",
        "--points_per_side",
        "8",
        "--binning",
        "4",
        "--min_area_fraction",
        "0.001",
        "--beam_threshold",
        "10",
        "--hole_threshold",
        "100",
        "--save_summary",
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
                "module load EM/cryoem-services\n" + " ".join(segment_command)
            ),
            "job": {
                "cpus_per_task": 1,
                "current_working_directory": f"{tmp_path}/output",
                "standard_output": f"{tmp_path}/output/micrograph_mask.out",
                "standard_error": f"{tmp_path}/output/micrograph_mask.err",
                "environment": ["USER=user", "HOME=/home"],
                "name": "SegmentHole",
                "nodes": "1",
                "partition": "partition",
                "prefer": "preference",
                "tasks": 1,
                "memory_per_node": {"number": 10000, "set": True, "infinite": False},
                "time_limit": {"number": 60, "set": True, "infinite": False},
                "tres_per_job": "gres/gpu:1",
            },
        },
    )
    mock_requests.Session().get.assert_called_with(
        url="/url/of/slurm/restapi/slurm/v0.0.40/job/1"
    )

    # Check that the correct messages were sent
    offline_transport.send.assert_called_once_with(
        "murfey_feedback",
        {
            "register": "segmented_hole",
            "micrograph": "/path/to/micrograph.mrc",
            "hole_percent": 95.5,
            "hole_detected": True,
            "beam_edge_detected": True,
        },
    )
