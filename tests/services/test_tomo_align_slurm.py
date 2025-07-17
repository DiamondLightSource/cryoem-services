from __future__ import annotations

import os
import sys
from unittest import mock

import pytest
from requests import Response
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import tomo_align_slurm
from tests.test_utils.config import cluster_submission_configuration


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.util.slurm_submission.subprocess.run")
@mock.patch("cryoemservices.util.slurm_submission.requests")
@mock.patch("cryoemservices.services.tomo_align.mrcfile")
@mock.patch("cryoemservices.services.tomo_align_slurm.transfer_files")
@mock.patch("cryoemservices.services.tomo_align_slurm.retrieve_files")
def test_tomo_align_slurm_service(
    mock_retrieve,
    mock_transfer,
    mock_mrcfile,
    mock_requests,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign for the slurm submission version
    This should call the mock subprocess then send messages on to
    the denoising, ispyb_connector and images services.
    """
    mock_subprocess().returncode = 0
    # Set up the returned job number
    response_object = Response()
    response_object._content = (
        '{"job_id": 1, "error_code": 0, "error": "", "jobs": [{"job_state": ["COMPLETED"]}]}'
    ).encode("utf8")
    response_object.status_code = 200
    mock_requests.Session().post.return_value = response_object
    mock_requests.Session().get.return_value = response_object

    mock_mrcfile.open().__enter__().header = {"nx": 2000, "ny": 3000}

    mock_transfer.return_value = ["test_stack.mrc", "angles.txt"]

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.mrc",
        "path_pattern": None,
        "input_file_list": str(
            [[f"{tmp_path}/MotionCorr/job002/Movies/input_file_1.mrc", "1.00"]]
        ),
        "vol_z": 1200,
        "align": None,
        "out_bin": 4,
        "tilt_axis": 85.0,
        "tilt_cor": 1,
        "flip_int": None,
        "flip_vol": 1,
        "wbp": None,
        "roi_file": None,
        "patch": None,
        "kv": None,
        "dose_per_frame": None,
        "frame_count": None,
        "align_file": None,
        "angle_file": f"{tmp_path}/angles.file",
        "align_z": None,
        "pixel_size": 1,
        "refine_flag": 1,
        "out_imod": 1,
        "out_imod_xf": None,
        "dark_tol": None,
        "manual_tilt_offset": None,
        "tomogram_uuid": 0,
        "relion_options": {},
    }

    # Construct the file which contains rest api submission information
    os.environ["ARETOMO2_EXECUTABLE"] = "slurm_AreTomo"
    os.environ["EXTRA_LIBRARIES"] = "/lib/aretomo"
    cluster_submission_configuration(tmp_path)

    # Set up the mock service
    service = tomo_align_slurm.TomoAlignSlurm(
        environment={
            "config": f"{tmp_path}/config.yaml",
            "slurm_cluster": "default",
            "queue": "",
        },
        transport=offline_transport,
    )
    service.initializing()

    service.rot_centre_z_list = [1.1, 2.1]
    service.tilt_offset = 1.1
    service.alignment_quality = 0.5

    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo_Imod").mkdir(
        parents=True
    )
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo_Imod/tilt.com").touch()
    with open(tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w") as aln_file:
        aln_file.write("dummy 0 1000 1.2 2.3 5 6 7 8 4.5")

    # Touch the expected output files
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc").touch()
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc.out").touch()
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc.err").touch()

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    # Check the newstack was run
    mock_subprocess.assert_any_call(
        [
            "newstack",
            "-fileinlist",
            f"{tmp_path}/Tomograms/job006/tomograms/test_stack_newstack.txt",
            "-output",
            f"{tmp_path}/Tomograms/job006/tomograms/test_stack.mrc",
            "-quiet",
        ],
        capture_output=True,
    )

    # Check the angle file
    assert (
        tmp_path / "Tomograms/job006/tomograms/test_stack_tilt_angles.txt"
    ).is_file()
    with open(
        tmp_path / "Tomograms/job006/tomograms/test_stack_tilt_angles.txt", "r"
    ) as angfile:
        angles_data = angfile.read()
    assert angles_data == "1.00  0\n"

    # Command which should run
    aretomo_command = [
        "slurm_AreTomo",
        "-InMrc",
        "test_stack.mrc",
        "-OutMrc",
        f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
        "-AngFile",
        f"{tmp_path}/Tomograms/job006/tomograms/test_stack_tilt_angles.txt",
        "-TiltCor",
        "1",
        "-VolZ",
        str(tomo_align_test_message["vol_z"]),
        "-TiltAxis",
        str(tomo_align_test_message["tilt_axis"]),
        "1",
        "-PixSize",
        "1.0",
        "-OutBin",
        str(tomo_align_test_message["out_bin"]),
        "-FlipVol",
        str(tomo_align_test_message["flip_vol"]),
        "-OutImod",
        str(tomo_align_test_message["out_imod"]),
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
                "export LD_LIBRARY_PATH=/lib/aretomo:$LD_LIBRARY_PATH\n"
                + " ".join(aretomo_command)
            ),
            "job": {
                "cpus_per_task": 1,
                "current_working_directory": f"{tmp_path}/Tomograms/job006/tomograms",
                "standard_output": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.out",
                "standard_error": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.err",
                "environment": ["USER=user", "HOME=/home"],
                "name": "AreTomo2",
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
        [
            tmp_path / "Tomograms/job006/tomograms/test_stack.mrc",
            tmp_path / "Tomograms/job006/tomograms/test_stack_tilt_angles.txt",
        ]
    )
    assert mock_retrieve.call_count == 1
    mock_retrieve.assert_any_call(
        job_directory=tmp_path / "Tomograms/job006/tomograms",
        files_to_skip=[
            tmp_path / "Tomograms/job006/tomograms/test_stack.mrc",
            tmp_path / "Tomograms/job006/tomograms/test_stack_tilt_angles.txt",
        ],
        basepath="test_stack",
    )
    assert mock_subprocess.call_count == 2


def test_parse_tomo_align_output(offline_transport, tmp_path):
    """
    Send test lines to the output parser
    to check the rotations and offsets are being read in
    """
    service = tomo_align_slurm.TomoAlignSlurm(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()

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


def test_transfer_files(tmp_path):
    """Test that existing files can be transferred, and non-existant files are not"""
    (tmp_path / "to_transfer").mkdir()
    (tmp_path / "to_transfer/file_exists").touch()
    transferred_files = tomo_align_slurm.transfer_files(
        [
            tmp_path / "to_transfer/file_exists",
            tmp_path / "to_transfer/file_does_not_exist",
        ],
        local_base=f"{tmp_path}/to_transfer",
        remote_base=f"{tmp_path}/destination",
    )

    assert transferred_files == [tmp_path / "to_transfer/file_exists"]
    assert (tmp_path / "destination/file_exists").is_file()
    assert not (tmp_path / "destination/file_does_not_exist").exists()


def test_retrieve_files(tmp_path):
    (tmp_path / "remote_system/job_dir/file_imod_dir").mkdir(parents=True)
    (tmp_path / "remote_system/job_dir/file_to_retrieve").touch()
    (tmp_path / "remote_system/job_dir/file_to_ignore").touch()
    (tmp_path / "remote_system/job_dir/different_basepath").touch()
    (tmp_path / "remote_system/job_dir/file_imod_dir/imod_file").touch()

    tomo_align_slurm.retrieve_files(
        job_directory=tmp_path / "local_system/job_dir",
        files_to_skip=[
            tmp_path / "local_system/job_dir/file_to_ignore",
            tmp_path / "local_system/job_dir/file_not_exists",
        ],
        basepath="file",
        local_base=f"{tmp_path}/local_system",
        remote_base=f"{tmp_path}/remote_system",
    )

    # File which should have been copied and removed
    assert (tmp_path / "local_system/job_dir/file_to_retrieve").is_file()
    assert not (tmp_path / "remote_system/job_dir/file_to_retrieve").exists()

    # File should have been ignored but removed anyway
    assert not (tmp_path / "local_system/job_dir/file_to_ignore").exists()
    assert not (tmp_path / "remote_system/job_dir/file_to_ignore").exists()

    # File in subfolder which should have been copied and removed
    assert (tmp_path / "local_system/job_dir/file_imod_dir/imod_file").is_file()
    assert not (tmp_path / "remote_system/job_dir/file_imod_dir").exists()

    # File with different basepath should have been left where it is
    assert not (tmp_path / "local_system/job_dir/different_basepath").exists()
    assert (tmp_path / "remote_system/job_dir/different_basepath").is_file()

    # File which doesn't exist
    assert not (tmp_path / "local_system/job_dir/file_not_exists").exists()
    assert not (tmp_path / "remote_system/job_dir/file_not_exists").exists()
