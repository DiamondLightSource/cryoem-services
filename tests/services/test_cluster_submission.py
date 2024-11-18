from __future__ import annotations

import json
import os
import sys
from unittest import mock

import pytest
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
        cf.write(f"slurm_credentials: {tmp_path}/slurm_credentials.yaml\n")
    os.environ["CRYOEMSERVICES_CONFIG"] = str(config_file)
    os.environ["USER"] = "user"

    # Create a slurm credentials file
    with open(tmp_path / "slurm_credentials.yaml", "w") as slurm_creds:
        slurm_creds.write("url: /slurm/url\n")
        slurm_creds.write("api_version: v0.0.40\n")
        slurm_creds.write("user: user\n")
        slurm_creds.write("user_token: /path/to/token.txt\n")


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("workflows.recipe.RecipeWrapper")
@mock.patch("cryoemservices.services.cluster_submission.slurm")
def test_cluster_submission_recipeless(
    mock_restapi, mock_rw, offline_transport, tmp_path
):
    """
    Send a test message to ClusterSubmission without any recipe information
    """
    cluster_submission_configuration(tmp_path)

    # Set up the returned job number
    mock_restapi.SlurmRestApi().submit_job().error = ""
    mock_restapi.SlurmRestApi().submit_job().job_id = 1

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
                "partition": "part",
                "prefer": "preferred_part",
                "scheduler": "slurm",
                "tasks": 2,
                "time_limit": 300,
            },
        }
    }

    # Set up the mock service
    service = cluster_submission.ClusterSubmission()
    service.transport = offline_transport
    service.start()
    service.run_submit_job(mock_rw, header=header, message={})

    # Check the calls to the job setup and submission
    mock_restapi.SlurmRestApi.assert_called_with(
        url="/slurm/url",
        version="v0.0.40",
        user_name="user",
        user_token="/path/to/token.txt",
    )
    mock_restapi.models.Uint64NoVal.assert_any_call(number=10, set=True)
    mock_restapi.models.Uint64NoVal.assert_called_with(number=20, set=True)
    mock_restapi.models.Uint32NoVal.assert_called_with(number=5, set=True)
    mock_restapi.models.JobDescMsg.assert_called_with(
        cpus_per_task=3,
        current_working_directory=str(tmp_path),
        environment=["USER=user"],
        memory_per_cpu=mock_restapi.models.Uint64NoVal(),
        memory_per_node=mock_restapi.models.Uint64NoVal(),
        name="test_job",
        nodes="1",
        partition="part",
        prefer="preferred_part",
        tasks=2,
        time_limit=mock_restapi.models.Uint32NoVal(),
        tres_per_node="gres/gpu:4",
        tres_per_job="gres/gpu:4",
    )
    mock_restapi.models.JobSubmitReq.assert_called_with(
        script="#!/bin/bash\n. /etc/profile.d/modules.sh\nsrun job",
        job=mock_restapi.models.JobDescMsg(),
    )
    mock_restapi.SlurmRestApi().submit_job.assert_called_with(
        mock_restapi.models.JobSubmitReq()
    )

    # Check the service registered success
    mock_rw.set_default_channel.assert_called_with("job_submitted")
    mock_rw.send.assert_called_with({"jobid": 1}, transaction=mock.ANY)


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("workflows.recipe.RecipeWrapper")
@mock.patch("cryoemservices.services.cluster_submission.slurm")
def test_cluster_submission_recipefile(
    mock_restapi, mock_rw, offline_transport, tmp_path
):
    """
    Send a test message to ClusterSubmission with a recipefile set
    """
    cluster_submission_configuration(tmp_path)

    # Set up the returned job number
    mock_restapi.SlurmRestApi().submit_job().error = ""
    mock_restapi.SlurmRestApi().submit_job().job_id = 1

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    mock_rw.recipe.pretty.return_value = "recipe example"
    mock_rw.recipe_step = {
        "parameters": {
            "recipefile": str(tmp_path / "recipefile"),
            "workingdir": str(tmp_path),
            "cluster": {
                "commands": "srun $RECIPEFILE",
                "cpus_per_task": 3,
                "job_name": "test_job",
                "nodes": 1,
                "partition": "part",
                "prefer": "preferred_part",
                "scheduler": "slurm",
                "tasks": 2,
            },
        }
    }

    # Set up the mock service
    service = cluster_submission.ClusterSubmission()
    service.transport = offline_transport
    service.start()
    service.run_submit_job(mock_rw, header=header, message={})

    # Check the calls to the job setup and submission
    assert (tmp_path / "recipefile").is_file()
    mock_rw.recipe.pretty.assert_called()
    mock_restapi.models.JobSubmitReq.assert_called_with(
        script=f"#!/bin/bash\n. /etc/profile.d/modules.sh\nsrun {tmp_path}/recipefile",
        job=mock_restapi.models.JobDescMsg(),
    )

    # Check the service registered success
    mock_rw.set_default_channel.assert_called_with("job_submitted")
    mock_rw.send.assert_called_with({"jobid": 1}, transaction=mock.ANY)


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("workflows.recipe.RecipeWrapper")
@mock.patch("cryoemservices.services.cluster_submission.slurm")
def test_cluster_submission_recipeenvironment(
    mock_restapi, mock_rw, offline_transport, tmp_path
):
    """
    Send a test message to ClusterSubmission with a recipeenvironment set
    """
    cluster_submission_configuration(tmp_path)

    # Set up the returned job number
    mock_restapi.SlurmRestApi().submit_job().error = ""
    mock_restapi.SlurmRestApi().submit_job().job_id = 1

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    mock_rw.environment = {"env": "env"}
    mock_rw.recipe_step = {
        "parameters": {
            "recipeenvironment": str(tmp_path / "recipe_env"),
            "workingdir": str(tmp_path),
            "cluster": {
                "commands": "srun job $RECIPEENV",
                "cpus_per_task": 3,
                "job_name": "test_job",
                "nodes": 1,
                "partition": "part",
                "prefer": "preferred_part",
                "scheduler": "slurm",
                "tasks": 2,
            },
        }
    }

    # Set up the mock service
    service = cluster_submission.ClusterSubmission()
    service.transport = offline_transport
    service.start()
    service.run_submit_job(mock_rw, header=header, message={})

    # Check the calls to the job setup and submission
    assert (tmp_path / "recipe_env").is_file()
    mock_restapi.models.JobSubmitReq.assert_called_with(
        script=f"#!/bin/bash\n. /etc/profile.d/modules.sh\nsrun job {tmp_path}/recipe_env",
        job=mock_restapi.models.JobDescMsg(),
    )

    # Check the service registered success
    mock_rw.set_default_channel.assert_called_with("job_submitted")
    mock_rw.send.assert_called_with({"jobid": 1}, transaction=mock.ANY)


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("workflows.recipe.RecipeWrapper")
@mock.patch("cryoemservices.services.cluster_submission.slurm")
def test_cluster_submission_recipewrapper(
    mock_restapi, mock_rw, offline_transport, tmp_path
):
    """
    Send a test message to ClusterSubmission with a recipewrapper set
    """
    cluster_submission_configuration(tmp_path)

    # Set up the returned job number
    mock_restapi.SlurmRestApi().submit_job().error = ""
    mock_restapi.SlurmRestApi().submit_job().job_id = 1

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    mock_rw.recipe_step = {
        "parameters": {
            "recipewrapper": str(tmp_path / "recipe_wrapper"),
            "workingdir": str(tmp_path),
            "cluster": {
                "commands": "srun job $RECIPEWRAP",
                "cpus_per_task": 3,
                "job_name": "test_job",
                "nodes": 1,
                "partition": "part",
                "prefer": "preferred_part",
                "scheduler": "slurm",
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
    service = cluster_submission.ClusterSubmission()
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

    mock_restapi.models.JobSubmitReq.assert_called_with(
        script=f"#!/bin/bash\n. /etc/profile.d/modules.sh\nsrun job {tmp_path}/recipe_wrapper",
        job=mock_restapi.models.JobDescMsg(),
    )

    # Check the service registered success
    mock_rw.set_default_channel.assert_called_with("job_submitted")
    mock_rw.send.assert_called_with({"jobid": 1}, transaction=mock.ANY)
