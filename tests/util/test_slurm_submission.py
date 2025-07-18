from __future__ import annotations

from unittest import mock

from requests import Response

from cryoemservices.util import slurm_submission
from cryoemservices.util.config import config_from_file
from tests.test_utils.config import cluster_submission_configuration


@mock.patch("cryoemservices.util.slurm_submission.requests")
def test_wait_for_job_completion_success(mock_requests, tmp_path):
    # Set up the returned job number
    response_object = Response()
    response_object._content = (
        '{"job_id": 1, "error_code": 0, "error": "", "jobs": [{"job_state": ["COMPLETED"]}]}'
    ).encode("utf8")
    response_object.status_code = 200
    mock_requests.Session().post.return_value = response_object
    mock_requests.Session().get.return_value = response_object

    # Construct the file which contains rest api submission information
    cluster_submission_configuration(tmp_path)

    service_config = config_from_file(tmp_path / "config.yaml")

    returned_job_state = slurm_submission.wait_for_job_completion(
        job_id=1,
        logger=mock.Mock(),
        service_config=service_config,
        cluster_name="default",
    )

    assert returned_job_state == "COMPLETED"

    # Check the slurm commands were run
    mock_requests.Session.assert_called()
    mock_requests.Session().headers.__setitem__.assert_any_call(
        "X-SLURM-USER-NAME", "user"
    )
    mock_requests.Session().headers.__setitem__.assert_any_call(
        "X-SLURM-USER-TOKEN", "token_key"
    )
    mock_requests.Session().get.assert_called_with(
        url="/url/of/slurm/restapi/slurm/v0.0.40/job/1"
    )
    mock_requests.Session().delete.assert_not_called()


@mock.patch("cryoemservices.util.slurm_submission.requests")
def test_wait_for_job_completion_timeout(mock_requests, tmp_path):
    # Set up the returned job number
    response_object = Response()
    response_object._content = (
        '{"job_id": 1, "error_code": 0, "error": "", "jobs": [{"job_state": ["RUNNING"]}]}'
    ).encode("utf8")
    response_object.status_code = 200
    mock_requests.Session().post.return_value = response_object
    mock_requests.Session().get.return_value = response_object

    # Construct the file which contains rest api submission information
    cluster_submission_configuration(tmp_path)

    service_config = config_from_file(tmp_path / "config.yaml")

    returned_job_state = slurm_submission.wait_for_job_completion(
        job_id=1,
        logger=mock.Mock(),
        service_config=service_config,
        cluster_name="default",
        timeout_counter=1,
    )

    assert returned_job_state == "CANCELLED"

    # Check the slurm commands were run
    mock_requests.Session.assert_called()
    mock_requests.Session().headers.__setitem__.assert_any_call(
        "X-SLURM-USER-NAME", "user"
    )
    mock_requests.Session().headers.__setitem__.assert_any_call(
        "X-SLURM-USER-TOKEN", "token_key"
    )
    mock_requests.Session().get.assert_called_with(
        url="/url/of/slurm/restapi/slurm/v0.0.40/job/1"
    )
    mock_requests.Session().delete.assert_called_with(
        url="/url/of/slurm/restapi/slurm/v0.0.40/job/1"
    )
