from __future__ import annotations

import os
import sys
from unittest import mock

import pytest
import zocalo.configuration
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import tomo_align_slurm


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
@mock.patch("cryoemservices.services.tomo_align_slurm.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.px.scatter")
@mock.patch("cryoemservices.services.tomo_align_slurm.transfer_files")
@mock.patch("cryoemservices.services.tomo_align_slurm.retrieve_files")
def test_tomo_align_slurm_service(
    mock_retrieve,
    mock_transfer,
    mock_plotly,
    mock_subprocess,
    mock_environment,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign for the slurm submission version
    This should call the mock subprocess then send messages on to
    the denoising, ispyb_connector and images services.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = (
        '{"job_id": "1", "jobs": [{"job_state": ["COMPLETED"]}]}'.encode("ascii")
    )
    mock_subprocess().stderr = "stderr".encode("ascii")

    mock_transfer.return_value = 0

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "parameters": {
            "stack_file": f"{tmp_path}/test_stack.st",
            "path_pattern": None,
            "input_file_list": str([[f"{tmp_path}/input_file_1.mrc", "1.00"]]),
            "position": None,
            "aretomo_output_file": f"{tmp_path}/test_stack_aretomo.mrc",
            "vol_z": 1200,
            "align": None,
            "out_bin": 4,
            "tilt_axis": None,
            "tilt_cor": 1,
            "flip_int": None,
            "flip_vol": 1,
            "wbp": None,
            "roi_file": [],
            "patch": None,
            "kv": None,
            "align_file": None,
            "angle_file": f"{tmp_path}/angles.file",
            "align_z": None,
            "pix_size": 1e-10,
            "init_val": None,
            "refine_flag": None,
            "out_imod": 1,
            "out_imod_xf": None,
            "dark_tol": None,
            "manual_tilt_offset": None,
        },
        "content": "dummy",
    }

    # Set up the mock service
    service = tomo_align_slurm.TomoAlignSlurm(environment=mock_environment)
    service.transport = offline_transport
    service.start()

    service.rot_centre_z_list = [1.1, 2.1]
    service.tilt_offset = 1.1
    service.alignment_quality = 0.5
    service.mag = 1000
    service.rot = 0

    (tmp_path / "test_stack_aretomo_Imod").mkdir()
    (tmp_path / "test_stack.aln").touch()
    (tmp_path / "test_stack_aretomo_Imod/tilt.com").touch()

    # Construct the file which contains rest api submission information
    os.environ["ARETOMO_EXECUTABLE"] = "slurm_AreTomo"
    os.environ["EXTRA_LIBRARIES"] = "/lib/aretomo"
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
    (tmp_path / "test_stack_aretomo.mrc.out").touch()
    (tmp_path / "test_stack_aretomo.mrc.err").touch()

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    # Check the newstack was run
    mock_subprocess.assert_any_call(
        [
            "newstack",
            "-fileinlist",
            f"{tmp_path}/test_stack_newstack.txt",
            "-output",
            f"{tmp_path}/test_stack.st",
            "-quiet",
        ]
    )

    # Check the slurm commands were run
    slurm_submit_command = (
        f'curl -H "X-SLURM-USER-NAME:user" -H "X-SLURM-USER-TOKEN:token_key" '
        '-H "Content-Type: application/json" -X POST '
        "/url/of/slurm/restapi/slurm/v0.0.40/job/submit "
        f"-d @{tmp_path}/test_stack_aretomo.mrc.json"
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
    assert mock_transfer.call_count == 1
    mock_transfer.assert_any_call([f"{tmp_path}/test_stack.st"])
    assert mock_retrieve.call_count == 1
    mock_retrieve.assert_any_call(
        job_directory=tmp_path,
        files_to_skip=[f"{tmp_path}/test_stack.st"],
        basepath="test_stack",
    )
    assert mock_plotly.call_count == 1
    assert mock_subprocess.call_count == 6


def test_parse_tomo_align_output(mock_environment, offline_transport, tmp_path):
    """
    Send test lines to the output parser
    to check the rotations and offsets are being read in
    """
    service = tomo_align_slurm.TomoAlignSlurm(environment=mock_environment)
    service.transport = offline_transport
    service.start()

    with open(tmp_path / "tomo_output.txt", "w") as tomo_output:
        tomo_output.write(
            "Rot center Z 100.0 200.0 300.0\n"
            "Rot center Z 150.0 250.0 350.0\n"
            "Tilt offset 1.0, CC: 0.5\n"
            "Best tilt axis:   57, Score:   0.07568\n"
        )

    tomo_align_slurm.TomoAlignSlurm.parse_tomo_output(
        service, str(tmp_path / "tomo_output.txt")
    )
    assert service.rot_centre_z_list == ["300.0", "350.0"]
    assert service.tilt_offset == 1.0
    assert service.alignment_quality == 0.07568
