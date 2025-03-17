from __future__ import annotations

import json
import os
import sys
from unittest import mock

import pytest
from requests import Response
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import cluster_submission


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


def cluster_submission_configuration(tmp_path):
    # Create a config file
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as cf:
        cf.write("rabbitmq_credentials: rmq_creds\n")
        cf.write(f"recipe_directory: {tmp_path}/recipes\n")
        cf.write("slurm_credentials:\n")
        cf.write(f"  default: {tmp_path}/slurm_credentials.yaml\n")
        cf.write(f"  extra: {tmp_path}/slurm_credentials_extra.yaml\n")
    os.environ["USER"] = "user"

    # Create dummy slurm credentials files
    with open(tmp_path / "slurm_credentials.yaml", "w") as slurm_creds:
        slurm_creds.write("url: /slurm/url\n")
        slurm_creds.write("api_version: v0.0.40\n")
        slurm_creds.write("user: user\n")
        slurm_creds.write(f"user_token: {tmp_path}/token.txt\n")
        slurm_creds.write("partition: part\n")
        slurm_creds.write("partition_preference: preferred_part\n")
    with open(tmp_path / "slurm_credentials_extra.yaml", "w") as slurm_creds:
        slurm_creds.write("url: /slurm/extra/url\n")
        slurm_creds.write("api_version: v0.0.41\n")
        slurm_creds.write("user: user2\n")
        slurm_creds.write(f"user_token: {tmp_path}/token_user2.txt\n")
        slurm_creds.write("partition: part\n")
        slurm_creds.write("partition_preference: preferred_part\n")

    with open(tmp_path / "token.txt", "w") as token_file:
        token_file.write("token")
    with open(tmp_path / "token_user2.txt", "w") as token_file:
        token_file.write("token2")


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("workflows.recipe.RecipeWrapper")
@mock.patch("cryoemservices.services.cluster_submission.requests")
def test_cluster_submission_recipeless(
    mock_requests, mock_rw, offline_transport, tmp_path
):
    """
    Send a test message to ClusterSubmission without any recipe information
    """
    cluster_submission_configuration(tmp_path)

    # Set up the returned job number
    response_object = Response()
    response_object._content = (
        '{"job_id": 1, "step_id": "0", "error_code": 0, "error": "", "job_submit_user_msg": "message"}'
    ).encode("utf8")
    response_object.status_code = 200
    mock_requests.Session().post.return_value = response_object

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    mock_rw.recipe_step = {
        "parameters": {
            "workingdir": str(tmp_path),
            "cluster": {
                "commands": "srun job",
                "cpus_per_task": 3,
                "gpus": 4,
                "gpus_per_node": 4,
                "job_name": "test_job",
                "memory_per_node": 20,
                "min_memory_per_cpu": 10,
                "nodes": 1,
                "tasks": 2,
                "time_limit": 300,
            },
        }
    }

    # Set up the mock service
    service = cluster_submission.ClusterSubmission(
        environment={
            "config": f"{tmp_path}/config.yaml",
            "slurm_cluster": "default",
            "queue": "",
        }
    )
    service.transport = offline_transport
    service.start()
    service.run_submit_job(mock_rw, header=header, message={})

    # Check the calls to the job setup and submission
    mock_requests.Session.assert_called()
    mock_requests.Session().headers.__setitem__.assert_any_call(
        "X-SLURM-USER-NAME", "user"
    )
    mock_requests.Session().headers.__setitem__.assert_any_call(
        "X-SLURM-USER-TOKEN", "token"
    )
    mock_requests.Session().post.assert_called_with(
        url="/slurm/url/slurm/v0.0.40/job/submit",
        json={
            "job": {
                "cpus_per_task": 3,
                "current_working_directory": str(tmp_path),
                "environment": ["USER=user"],
                "name": "test_job",
                "nodes": "1",
                "partition": "part",
                "prefer": "preferred_part",
                "tasks": 2,
                "memory_per_cpu": {"set": True, "infinite": False, "number": 10},
                "memory_per_node": {"set": True, "infinite": False, "number": 20},
                "time_limit": {"set": True, "infinite": False, "number": 5},
                "tres_per_node": "gres/gpu:4",
                "tres_per_job": "gres/gpu:4",
            },
            "script": "#!/bin/bash\n. /etc/profile.d/modules.sh\nsrun job",
        },
    )

    # Check the service registered success
    mock_rw.set_default_channel.assert_called_with("job_submitted")
    mock_rw.send.assert_called_with({"jobid": 1}, transaction=mock.ANY)


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("workflows.recipe.RecipeWrapper")
@mock.patch("cryoemservices.services.cluster_submission.requests")
def test_cluster_submission_wrapper(
    mock_requests, mock_rw, offline_transport, tmp_path
):
    """
    Send a test message to ClusterSubmission with a recipewrapper set
    """
    cluster_submission_configuration(tmp_path)

    # Set up the returned job number
    response_object = Response()
    response_object._content = (
        '{"job_id": 1, "step_id": "0", "error_code": 0, "error": "", "job_submit_user_msg": "message"}'
    ).encode("utf8")
    response_object.status_code = 200
    mock_requests.Session().post.return_value = response_object

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    mock_rw.recipe_step = {
        "parameters": {
            "wrapper": str(tmp_path / "recipe_wrapper"),
            "workingdir": str(tmp_path),
            "cluster": {
                "commands": "srun job $RECIPEWRAP",
                "cpus_per_task": 3,
                "job_name": "test_job",
                "nodes": 1,
                "tasks": 2,
            },
        }
    }

    mock_rw.environment = {"env": "env"}
    mock_rw.recipe_pointer = 1
    mock_rw.recipe.recipe = "recipe_name"
    mock_rw.recipe_path = "/path/to/recipe"
    mock_rw.payload = "payload"

    # Set up the mock service
    service = cluster_submission.ClusterSubmission(
        environment={
            "config": f"{tmp_path}/config.yaml",
            "slurm_cluster": "default",
            "queue": "",
        }
    )
    service.transport = offline_transport
    service.start()
    service.run_submit_job(mock_rw, header=header, message={})

    # Check the calls to the job setup and submission
    assert (tmp_path / "recipe_wrapper").is_file()
    with open(tmp_path / "recipe_wrapper", "r") as wrapper_file:
        output_json = json.loads(wrapper_file.read())
    assert output_json["recipe"] == "recipe_name"
    assert output_json["recipe-pointer"] == 1
    assert output_json["environment"] == {"env": "env"}
    assert output_json["recipe-path"] == "/path/to/recipe"
    assert output_json["payload"] == "payload"

    mock_requests.Session().post.assert_called_with(
        url="/slurm/url/slurm/v0.0.40/job/submit",
        json={
            "job": {
                "cpus_per_task": 3,
                "current_working_directory": str(tmp_path),
                "environment": ["USER=user"],
                "name": "test_job",
                "nodes": "1",
                "partition": "part",
                "prefer": "preferred_part",
                "tasks": 2,
            },
            "script": f"#!/bin/bash\n. /etc/profile.d/modules.sh\nsrun job {tmp_path}/recipe_wrapper",
        },
    )

    # Check the service registered success
    mock_rw.set_default_channel.assert_called_with("job_submitted")
    mock_rw.send.assert_called_with({"jobid": 1}, transaction=mock.ANY)


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("workflows.recipe.RecipeWrapper")
@mock.patch("cryoemservices.services.cluster_submission.requests")
def test_cluster_submission_extra_cluster(
    mock_requests, mock_rw, offline_transport, tmp_path
):
    """
    Send a test message to ClusterSubmission for the second configured cluster
    """
    cluster_submission_configuration(tmp_path)

    # Set up the returned job number
    response_object = Response()
    response_object._content = (
        '{"job_id": 1, "step_id": "0", "error_code": 0, "error": "", "job_submit_user_msg": "message"}'
    ).encode("utf8")
    response_object.status_code = 200
    mock_requests.Session().post.return_value = response_object

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    mock_rw.recipe_step = {
        "parameters": {
            "workingdir": str(tmp_path),
            "cluster": {
                "commands": "srun job",
                "cpus_per_task": 3,
                "gpus": 4,
                "gpus_per_node": 4,
                "job_name": "test_job",
                "memory_per_node": 20,
                "min_memory_per_cpu": 10,
                "nodes": 1,
                "tasks": 2,
                "time_limit": 300,
            },
        }
    }

    # Set up the mock service
    service = cluster_submission.ClusterSubmission(
        environment={
            "config": f"{tmp_path}/config.yaml",
            "slurm_cluster": "extra",
            "queue": "",
        }
    )
    service.transport = offline_transport
    service.start()
    service.run_submit_job(mock_rw, header=header, message={})

    # Check the calls to the job setup and submission
    mock_requests.Session().headers.__setitem__.assert_any_call(
        "X-SLURM-USER-NAME", "user2"
    )
    mock_requests.Session().headers.__setitem__.assert_any_call(
        "X-SLURM-USER-TOKEN", "token2"
    )
    mock_requests.Session().post.assert_called_with(
        url="/slurm/extra/url/slurm/v0.0.41/job/submit",
        json={
            "job": {
                "cpus_per_task": 3,
                "current_working_directory": str(tmp_path),
                "environment": ["USER=user"],
                "name": "test_job",
                "nodes": "1",
                "partition": "part",
                "prefer": "preferred_part",
                "tasks": 2,
                "memory_per_cpu": {"set": True, "infinite": False, "number": 10},
                "memory_per_node": {"set": True, "infinite": False, "number": 20},
                "time_limit": {"set": True, "infinite": False, "number": 5},
                "tres_per_node": "gres/gpu:4",
                "tres_per_job": "gres/gpu:4",
            },
            "script": "#!/bin/bash\n. /etc/profile.d/modules.sh\nsrun job",
        },
    )

    # Check the service registered success
    mock_rw.set_default_channel.assert_called_with("job_submitted")
    mock_rw.send.assert_called_with({"jobid": 1}, transaction=mock.ANY)
