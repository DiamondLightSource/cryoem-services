from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest
from requests import Response
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import icebreaker
from cryoemservices.util.relion_service_options import RelionServiceOptions
from tests.test_utils.config import cluster_submission_configuration

output_relion_options = dict(RelionServiceOptions())


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


def icebreaker_group_output_file(*args):
    input_file = Path(args[0])
    (input_file.parent / f"grouped/{input_file.stem}_grouped.mrc").touch()


def icebreaker_flatten_output_file(*args):
    input_file = Path(args[0])
    (input_file.parent / f"flattened/{input_file.stem}_flattened.mrc").touch()


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch(
    "cryoemservices.services.icebreaker.icebreaker_icegroups_multi.multigroup",
    side_effect=icebreaker_group_output_file,
)
def test_icebreaker_micrographs_service(mock_icebreaker, offline_transport, tmp_path):
    """
    Send a test message to IceBreaker for running the micrographs job
    This should call the mock subprocess
    then send a message on to the node_creator service.
    It also creates the icebreaker summary jobs.
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    icebreaker_test_message = {
        "icebreaker_type": "micrographs",
        "input_micrographs": f"{tmp_path}/MotionCorr/job002/Movies/sample.mrc",
        "input_particles": None,
        "output_path": f"{tmp_path}/IceBreaker/job003/",
        "cpus": 1,
        "mc_uuid": 0,
        "relion_options": {"do_icebreaker_jobs": "True"},
        "total_motion": 0.5,
        "early_motion": 0.2,
        "late_motion": 0.3,
    }

    # Set up the mock service and send a message to the service
    service = icebreaker.IceBreaker(
        environment={"queue": ""},
        rabbitmq_credentials=tmp_path,
    )
    service._transport = offline_transport
    service.initializing()
    service.icebreaker(None, header=header, message=icebreaker_test_message)

    # Check the correct icebreaker command was run
    assert mock_icebreaker.call_count == 1
    mock_icebreaker.assert_called_with(Path("IB_tmp_sample/sample.mrc"))

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        "icebreaker",
        {
            "icebreaker_type": "summary",
            "input_micrographs": f"{tmp_path}/IceBreaker/job003/Movies/sample_grouped.mrc",
            "mc_uuid": 0,
            "relion_options": output_relion_options,
            "output_path": f"{tmp_path}/IceBreaker/job005/",
        },
    )
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "icebreaker.micrograph_analysis.micrographs",
            "input_file": icebreaker_test_message["input_micrographs"],
            "output_file": icebreaker_test_message["output_path"],
            "relion_options": output_relion_options,
            "command": (
                "ib_job --j 1 --single_mic MotionCorr/job002/Movies/sample.mrc "
                f"--o {tmp_path}/IceBreaker/job003/ --mode group"
            ),
            "stdout": "",
            "stderr": "",
            "results": {
                "icebreaker_type": "micrographs",
                "total_motion": 0.5,
                "early_motion": 0.2,
                "late_motion": 0.3,
            },
            "success": True,
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch(
    "cryoemservices.services.icebreaker.icebreaker_equalize_multi.multigroup",
    side_effect=icebreaker_flatten_output_file,
)
def test_icebreaker_enhancecontrast_service(
    mock_icebreaker, offline_transport, tmp_path
):
    """
    Send a test message to IceBreaker for running the enhance contrast job
    This should call the mock subprocess
    then send a message on to the node_creator service
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    icebreaker_test_message = {
        "icebreaker_type": "enhancecontrast",
        "input_micrographs": f"{tmp_path}/MotionCorr/job002/Movies/sample.mrc",
        "input_particles": None,
        "output_path": f"{tmp_path}/IceBreaker/job004/",
        "cpus": 1,
        "mc_uuid": 0,
        "relion_options": {"options": "options"},
        "total_motion": 0.5,
        "early_motion": 0.2,
        "late_motion": 0.3,
    }

    # Set up the mock service and send a message to the service
    service = icebreaker.IceBreaker(
        environment={"queue": ""},
        rabbitmq_credentials=tmp_path,
    )
    service._transport = offline_transport
    service.initializing()
    service.icebreaker(None, header=header, message=icebreaker_test_message)

    # Check the correct icebreaker command was run
    assert mock_icebreaker.call_count == 1
    mock_icebreaker.assert_called_with(Path("IB_tmp_sample/sample.mrc"))

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "icebreaker.micrograph_analysis.enhancecontrast",
            "input_file": icebreaker_test_message["input_micrographs"],
            "output_file": icebreaker_test_message["output_path"],
            "relion_options": output_relion_options,
            "command": (
                "ib_job --j 1 --single_mic MotionCorr/job002/Movies/sample.mrc "
                f"--o {tmp_path}/IceBreaker/job004/ --mode flatten"
            ),
            "stdout": "",
            "stderr": "",
            "results": {
                "icebreaker_type": "enhancecontrast",
                "total_motion": 0.5,
                "early_motion": 0.2,
                "late_motion": 0.3,
            },
            "success": True,
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.icebreaker.single_mic_5fig")
def test_icebreaker_summary_service(mock_icebreaker, offline_transport, tmp_path):
    """
    Send a test message to IceBreaker for running the summary job
    This should call the mock subprocess
    then send a message on to the node_creator service
    """
    mock_icebreaker.return_value = "sample_grouped.star,0,1,2,3,4"

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    icebreaker_test_message = {
        "icebreaker_type": "summary",
        "input_micrographs": f"{tmp_path}/IceBreaker/job003/Movies/sample_grouped.star",
        "input_particles": None,
        "output_path": f"{tmp_path}/IceBreaker/job005/",
        "cpus": 1,
        "mc_uuid": 0,
        "relion_options": {"options": "options"},
        "total_motion": 0.5,
        "early_motion": 0.2,
        "late_motion": 0.3,
    }

    # Set up the mock service and send a message to the service
    service = icebreaker.IceBreaker(
        environment={"queue": ""},
        rabbitmq_credentials=tmp_path,
    )
    service._transport = offline_transport
    service.initializing()
    service.icebreaker(None, header=header, message=icebreaker_test_message)

    # Check the correct icebreaker command was run
    assert mock_icebreaker.call_count == 1
    mock_icebreaker.assert_called_with(
        f"{tmp_path}/IceBreaker/job005/IB_input/sample_grouped.star"
    )

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "icebreaker.micrograph_analysis.summary",
            "input_file": icebreaker_test_message["input_micrographs"],
            "output_file": icebreaker_test_message["output_path"],
            "relion_options": output_relion_options,
            "command": (
                "ib_5fig --j 1 "
                "--single_mic IceBreaker/job003/Movies/sample_grouped.star "
                f"--o {tmp_path}/IceBreaker/job005/"
            ),
            "stdout": "",
            "stderr": "",
            "results": {
                "icebreaker_type": "summary",
                "total_motion": 0.5,
                "early_motion": 0.2,
                "late_motion": 0.3,
                "summary": ["0", "1", "2", "3", "4"],
            },
            "success": True,
        },
    )
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "minimum": "0",
            "q1": "1",
            "median": "2",
            "q3": "3",
            "maximum": "4",
            "ispyb_command": "buffer",
            "buffer_lookup": {"motion_correction_id": 0},
            "buffer_command": {"ispyb_command": "insert_relative_ice_thickness"},
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.icebreaker.ice_groups.main")
def test_icebreaker_particles_service(mock_icebreaker, offline_transport, tmp_path):
    """
    Send a test message to IceBreaker for running the particle analysis job
    This should call the mock subprocess
    then send a message on to the node_creator service
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    icebreaker_test_message = {
        "icebreaker_type": "particles",
        "input_micrographs": f"{tmp_path}/IceBreaker/job003/Movies/sample_grouped.star",
        "input_particles": f"{tmp_path}/Select/job009/particles_split1.star",
        "output_path": f"{tmp_path}/IceBreaker/job011/",
        "cpus": 1,
        "mc_uuid": 0,
        "relion_options": {"options": "options"},
        "total_motion": 0.5,
        "early_motion": 0.2,
        "late_motion": 0.3,
        "submit_to_slurm": False,
    }

    # Set up the mock service and send a message to the service
    service = icebreaker.IceBreaker(
        environment={"queue": ""},
        rabbitmq_credentials=tmp_path,
    )
    service._transport = offline_transport
    service.initializing()
    service.icebreaker(None, header=header, message=icebreaker_test_message)

    # Check the correct icebreaker command was run and the starfile was made
    assert mock_icebreaker.call_count == 1
    mock_icebreaker.assert_called_with(
        f"{tmp_path}/Select/job009/particles_split1.star",
        f"{tmp_path}/IceBreaker/job003/Movies/sample_grouped.star",
    )
    assert Path(f"{tmp_path}/IceBreaker/job011/ib_icegroups.star").is_file()

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "icebreaker.micrograph_analysis.particles",
            "input_file": icebreaker_test_message["input_micrographs"]
            + ":"
            + icebreaker_test_message["input_particles"],
            "output_file": icebreaker_test_message["output_path"],
            "relion_options": output_relion_options,
            "command": (
                "ib_group --j 1 "
                "--in_mics IceBreaker/job003/Movies/sample_grouped.star "
                "--in_parts Select/job009/particles_split1.star "
                f"--o {tmp_path}/IceBreaker/job011/"
            ),
            "stdout": "",
            "stderr": "",
            "results": {
                "icebreaker_type": "particles",
                "total_motion": 0.5,
                "early_motion": 0.2,
                "late_motion": 0.3,
            },
            "success": True,
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.util.slurm_submission.requests")
def test_icebreaker_particles_service_slurm(mock_requests, offline_transport, tmp_path):
    """
    Send a test message to IceBreaker for running the particle analysis job
    using the slurm submission method
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
    icebreaker_test_message = {
        "icebreaker_type": "particles",
        "input_micrographs": f"{tmp_path}/IceBreaker/job003/Movies/sample_grouped.star",
        "input_particles": f"{tmp_path}/Select/job009/particles_split1.star",
        "output_path": f"{tmp_path}/IceBreaker/job011/",
        "cpus": 1,
        "mc_uuid": 0,
        "relion_options": {"options": "options"},
        "total_motion": 0.5,
        "early_motion": 0.2,
        "late_motion": 0.3,
        "submit_to_slurm": True,
    }

    # Construct the file which contains rest api submission information
    cluster_submission_configuration(tmp_path)

    # Set up the mock service and send a message to the service
    service = icebreaker.IceBreaker(
        environment={
            "config": f"{tmp_path}/config.yaml",
            "slurm_cluster": "default",
            "queue": "",
        },
        rabbitmq_credentials=tmp_path,
    )
    service._transport = offline_transport
    service.initializing()
    service.icebreaker(None, header=header, message=icebreaker_test_message)

    ib_command = [
        "ib_group",
        "--j",
        "1",
        "--in_mics",
        "IceBreaker/job003/Movies/sample_grouped.star",
        "--in_parts Select/job009/particles_split1.star",
        f"--o {tmp_path}/IceBreaker/job011/",
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
                "module load EM/icebreaker/dev\n" + " ".join(ib_command)
            ),
            "job": {
                "cpus_per_task": 1,
                "current_working_directory": str(tmp_path),
                "standard_output": f"{tmp_path}/IceBreaker/job011/slurm.out",
                "standard_error": f"{tmp_path}/IceBreaker/job011/slurm.err",
                "environment": ["USER=user", "HOME=/home"],
                "name": "IceBreaker",
                "nodes": "1",
                "partition": "partition",
                "prefer": "preference",
                "tasks": 1,
                "memory_per_node": {"number": 1000, "set": True, "infinite": False},
                "time_limit": {"number": 60, "set": True, "infinite": False},
            },
        },
    )
    mock_requests.Session().get.assert_called_with(
        url="/url/of/slurm/restapi/slurm/v0.0.40/job/1"
    )
