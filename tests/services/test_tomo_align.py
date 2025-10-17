from __future__ import annotations

import json
import time
from subprocess import CompletedProcess
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import tomo_align
from cryoemservices.util.relion_service_options import RelionServiceOptions


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@mock.patch("cryoemservices.services.tomo_align.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.mrcfile")
def test_tomo_align_service_file_list(
    mock_mrcfile,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign
    This should call the mock subprocess then send messages on to
    the denoising, ispyb_connector and images services.
    """
    mock_mrcfile.open().__enter__().header = {"nx": 3000, "ny": 4000}

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.st",
        "path_pattern": None,
        "input_file_list": str(
            [
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_001_0.0.mrc", "1.00"],
            ]
        ),
        "vol_z": 1200,
        "align": None,
        "out_bin": 4,
        "tilt_axis": 90,
        "tilt_cor": 1,
        "flip_int": None,
        "flip_vol": 1,
        "flip_vol_post_reconstruction": False,
        "wbp": None,
        "roi_file": None,
        "patch": None,
        "kv": None,
        "dose_per_frame": None,
        "frame_count": None,
        "align_file": None,
        "align_z": None,
        "pixel_size": 1,
        "refine_flag": -1,
        "make_angle_file": True,
        "out_imod": 1,
        "out_imod_xf": None,
        "dark_tol": None,
        "manual_tilt_offset": None,
        "relion_options": {},
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["pixel_size"] = tomo_align_test_message["pixel_size"]
    output_relion_options["pixel_size_downscaled"] = (
        4 * tomo_align_test_message["pixel_size"]
    )
    output_relion_options["tomo_size_x"] = 3000
    output_relion_options["tomo_size_y"] = 4000
    output_relion_options["tilt_axis_angle"] = 86.0

    # Set up the mock service
    service = tomo_align.TomoAlign(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()

    def write_aretomo_outputs(command, capture_output):
        if command[0] != "AreTomo2":
            return CompletedProcess("", returncode=0)
        # Set up outputs: stack_Imod file like AreTomo2, no exclusions but with space
        (tmp_path / "Tomograms/job006/tomograms/test_stack_Imod").mkdir(parents=True)
        (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc").touch()
        with open(
            tmp_path / "Tomograms/job006/tomograms/test_stack_Imod/tilt.com", "w"
        ) as dark_file:
            dark_file.write("EXCLUDELIST ")
        with open(
            tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w"
        ) as aln_file:
            aln_file.write("dummy 86.0 1000 1.2 2.3 5 6 7 8 4.5")
        return CompletedProcess(
            "",
            returncode=0,
            stdout=(
                "Rot center Z 100.0 200.0 3.1\n"
                "Rot center Z 150.0 250.0 2.1\n"
                "Tilt offset 1.1, CC: 0.5\n"
                "Best tilt axis:   57, Score:   0.5\n"
            ).encode("ascii"),
            stderr="stderr".encode("ascii"),
        )

    mock_subprocess.side_effect = write_aretomo_outputs

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    aretomo_command = [
        "AreTomo2",
        "-InMrc",
        tomo_align_test_message["stack_file"],
        "-OutMrc",
        f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
        "-AngFile",
        f"{tmp_path}/Tomograms/job006/tomograms/test_stack_tilt_angles.txt",
        "-TiltCor",
        "1",
        "-VolZ",
        str(tomo_align_test_message["vol_z"]),
        "-TiltAxis",
        "90.0",
        "-1",
        "-PixSize",
        "1.0",
        "-OutBin",
        str(tomo_align_test_message["out_bin"]),
        "-FlipVol",
        str(tomo_align_test_message["flip_vol"]),
        "-OutImod",
        str(tomo_align_test_message["out_imod"]),
    ]

    # Check the expected calls were made
    assert mock_subprocess.call_count == 2
    mock_subprocess.assert_any_call(
        [
            "newstack",
            "-fileinlist",
            f"{tmp_path}/Tomograms/job006/tomograms/test_stack_newstack.txt",
            "-output",
            tomo_align_test_message["stack_file"],
            "-quiet",
        ],
        capture_output=True,
    )
    mock_subprocess.assert_any_call(
        aretomo_command,
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
    assert angles_data == "1.00  1\n"

    # Check the shift plot
    with open(
        tmp_path / "Tomograms/job006/tomograms/test_stack_xy_shift_plot.json"
    ) as shift_plot:
        shift_data = json.load(shift_plot)
    assert shift_data["data"][0]["x"] == [1.2]
    assert shift_data["data"][0]["y"] == [2.3]

    # Check that the correct messages were sent
    assert offline_transport.send.call_count == 12
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.excludetilts",
            "experiment_type": "tomography",
            "input_file": f"{tmp_path}/CtfFind/job003/Movies/Position_1_001_0.0.mrc",
            "output_file": f"{tmp_path}/ExcludeTiltImages/job004/tilts/Position_1_001_0.0.mrc",
            "relion_options": output_relion_options,
            "command": "",
            "stdout": "",
            "stderr": "",
            "success": True,
        },
    )
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.aligntiltseries.aretomo",
            "experiment_type": "tomography",
            "input_file": f"{tmp_path}/ExcludeTiltImages/job004/tilts/Position_1_001_0.0.mrc",
            "output_file": f"{tmp_path}/AlignTiltSeries/job005/tilts/Position_1_001_0.0.mrc",
            "relion_options": output_relion_options,
            "command": "",
            "stdout": "",
            "stderr": "",
            "results": {
                "TomoXTilt": "0.00",
                "TomoYTilt": "4.5",
                "TomoZRot": "86.0",
                "TomoXShiftAngst": "1.2",
                "TomoYShiftAngst": "2.3",
            },
            "success": True,
        },
    )
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "experiment_type": "tomography",
            "job_type": "relion.reconstructtomograms",
            "input_file": f"{tmp_path}/MotionCorr/job002/Movies/Position_1_001_0.0.mrc",
            "output_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
            "relion_options": output_relion_options | {"tilt_axis_angle": 85.0},
            "command": " ".join(aretomo_command),
            "stdout": (
                "Rot center Z 100.0 200.0 3.1\n"
                "Rot center Z 150.0 250.0 2.1\n"
                "Tilt offset 1.1, CC: 0.5\n"
                "Best tilt axis:   57, Score:   0.5\n"
            ),
            "stderr": "stderr",
            "success": True,
        },
    )
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "ispyb_command": "multipart_message",
            "ispyb_command_list": [
                {
                    "ispyb_command": "insert_tomogram",
                    "volume_file": "test_stack_aretomo.mrc",
                    "stack_file": tomo_align_test_message["stack_file"],
                    "size_x": 750,
                    "size_y": 1000,
                    "size_z": 300,
                    "pixel_spacing": "4.0",
                    "tilt_angle_offset": "1.1",
                    "z_shift": "2.1",
                    "file_directory": f"{tmp_path}/Tomograms/job006/tomograms",
                    "central_slice_image": "test_stack_aretomo_thumbnail.jpeg",
                    "tomogram_movie": "test_stack_aretomo_movie.png",
                    "xy_shift_plot": "test_stack_xy_shift_plot.json",
                    "proj_xy": "test_stack_aretomo_projXY.jpeg",
                    "proj_xz": "test_stack_aretomo_projXZ.jpeg",
                    "alignment_quality": "0.5",
                },
                {
                    "ispyb_command": "insert_tilt_image_alignment",
                    "psd_file": None,
                    "refined_magnification": "1000.0",
                    "refined_tilt_angle": "4.5",
                    "refined_tilt_axis": "86.0",
                    "path": f"{tmp_path}/MotionCorr/job002/Movies/Position_1_001_0.0.mrc",
                },
            ],
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_central_slice",
            "file": tomo_align_test_message["stack_file"],
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_to_apng",
            "file": tomo_align_test_message["stack_file"],
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_central_slice",
            "file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_to_apng",
            "file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_to_jpeg",
            "file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo_projXY.mrc",
            "pixel_spacing": "4.0",
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_to_jpeg",
            "file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo_projXZ.mrc",
            "pixel_spacing": "4.0",
        },
    )
    offline_transport.send.assert_any_call(
        "denoise",
        {
            "volume": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
            "output_dir": f"{tmp_path}/Denoise/job007/tomograms",
            "relion_options": output_relion_options,
        },
    )
    offline_transport.send.assert_any_call("success", {})


@mock.patch("cryoemservices.services.tomo_align.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.mrcfile")
def test_tomo_align_service_file_list_repeated_tilt(
    mock_mrcfile,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign with a duplicated tilt angle
    Only the newest one of the duplicated tilts should be used
    """
    mock_mrcfile.open().__enter__().header = {"nx": 3000, "ny": 4000}

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.st",
        "input_file_list": str(
            [
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_001_0.0.mrc", "1.00"],
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_002_0.0.mrc", "1.00"],
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_003_0.0.mrc", "1.00"],
            ]
        ),
        "pixel_size": 1,
        "relion_options": {},
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["pixel_size"] = 1
    output_relion_options["pixel_size_downscaled"] = 4
    output_relion_options["tomo_size_x"] = 3000
    output_relion_options["tomo_size_y"] = 4000

    # Create the input files. Needs sleeps to ensure distinct timestamps
    (tmp_path / "MotionCorr/job002/Movies").mkdir(parents=True)
    (tmp_path / "MotionCorr/job002/Movies/Position_1_001_0.0.mrc").touch()
    time.sleep(1)
    (tmp_path / "MotionCorr/job002/Movies/Position_1_002_0.0.mrc").touch()
    time.sleep(1)
    (tmp_path / "MotionCorr/job002/Movies/Position_1_003_0.0.mrc").touch()

    # Set up the mock service
    service = tomo_align.TomoAlign(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()

    def write_aretomo_outputs(command, capture_output):
        if command[0] != "AreTomo2":
            return CompletedProcess("", returncode=0)
        # Set up outputs: stack_Imod file like AreTomo2, no exclusions but with space
        (tmp_path / "Tomograms/job006/tomograms/test_stack_Imod").mkdir(parents=True)
        (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc").touch()
        with open(
            tmp_path / "Tomograms/job006/tomograms/test_stack_Imod/tilt.com", "w"
        ) as dark_file:
            dark_file.write("EXCLUDELIST ")
        with open(
            tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w"
        ) as aln_file:
            aln_file.write("dummy 0 1000 1.2 2.3 5 6 7 8 4.5")
        return CompletedProcess(
            "",
            returncode=0,
            stdout="stdout".encode("ascii"),
            stderr="stderr".encode("ascii"),
        )

    mock_subprocess.side_effect = write_aretomo_outputs

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    # Check the expected calls were made
    assert (
        tmp_path / "Tomograms/job006/tomograms/test_stack_xy_shift_plot.json"
    ).is_file()
    assert mock_subprocess.call_count == 3

    # This one runs the post-reconstruction volume flip
    mock_subprocess.assert_any_call(
        [
            "rotatevol",
            "-i",
            f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
            "-ou",
            f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
            "-size",
            "750,1000,300",
            "-a",
            "90,-90,0",
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
    assert angles_data == "1.00  3\n"

    # Check the stack file has only the last one of the duplicated tilt angles
    with open(
        tmp_path / "Tomograms/job006/tomograms/test_stack_newstack.txt", "r"
    ) as newstack_file:
        newstack_tilts = newstack_file.read()
    assert (
        newstack_tilts
        == f"1\n{tmp_path}/MotionCorr/job002/Movies/Position_1_003_0.0.mrc\n0\n"
    )

    # Look at a sample of the messages to check they use input file 3
    assert offline_transport.send.call_count == 12
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.excludetilts",
            "experiment_type": "tomography",
            "input_file": f"{tmp_path}/CtfFind/job003/Movies/Position_1_003_0.0.mrc",
            "output_file": f"{tmp_path}/ExcludeTiltImages/job004/tilts/Position_1_003_0.0.mrc",
            "relion_options": output_relion_options,
            "command": "",
            "stdout": "",
            "stderr": "",
            "success": True,
        },
    )
    offline_transport.send.assert_any_call("success", {})


@mock.patch("cryoemservices.services.tomo_align.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.mrcfile")
def test_tomo_align_service_file_list_zero_rotation(
    mock_mrcfile,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign with a tilt axis of zero to test rotation of volume
    """
    mock_mrcfile.open().__enter__().header = {"nx": 3000, "ny": 4000}

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.st",
        "input_file_list": str(
            [
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_001_0.0.mrc", "1.00"],
            ]
        ),
        "pixel_size": 1,
        "tilt_axis": 0,
        "relion_options": {},
    }

    # Create the input files. Needs sleeps to ensure distinct timestamps
    (tmp_path / "MotionCorr/job002/Movies").mkdir(parents=True)
    (tmp_path / "MotionCorr/job002/Movies/Position_1_001_0.0.mrc").touch()

    # Set up the mock service
    service = tomo_align.TomoAlign(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()

    def write_aretomo_outputs(command, capture_output):
        if command[0] != "AreTomo2":
            return CompletedProcess("", returncode=0)
        # Set up outputs: stack_Imod file like AreTomo2, no exclusions but with space
        (tmp_path / "Tomograms/job006/tomograms/test_stack_Imod").mkdir(parents=True)
        (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc").touch()
        (tmp_path / "Tomograms/job006/tomograms/test_stack_Imod/tilt.com").touch()
        with open(
            tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w"
        ) as aln_file:
            aln_file.write("dummy 0 1000 1.2 2.3 5 6 7 8 4.5")
        return CompletedProcess(
            "",
            returncode=0,
            stdout="stdout".encode("ascii"),
            stderr="stderr".encode("ascii"),
        )

    mock_subprocess.side_effect = write_aretomo_outputs

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    # Check the expected calls were made
    assert (
        tmp_path / "Tomograms/job006/tomograms/test_stack_xy_shift_plot.json"
    ).is_file()
    assert mock_subprocess.call_count == 3
    assert offline_transport.send.call_count == 12

    # This one runs the post-reconstruction volume flip
    mock_subprocess.assert_any_call(
        [
            "rotatevol",
            "-i",
            f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
            "-ou",
            f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
            "-size",
            "750,1000,300",
            "-a",
            "0,0,-90",
        ],
        capture_output=True,
    )


@mock.patch("cryoemservices.services.tomo_align.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.mrcfile")
def test_tomo_align_service_file_list_bad_tilts(
    mock_mrcfile,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign with a tilts with bad motion correction
    This tilt should be removed
    """
    mock_mrcfile.open().__enter__().header = {"nx": 3000, "ny": 4000}

    # Create the Relion star files with different motion model results
    (tmp_path / "MotionCorr/job002/Movies").mkdir(parents=True)
    with open(
        tmp_path / "MotionCorr/job002/Movies/Position_1_001_0.0.star", "w"
    ) as mc_star:
        # No local motion model
        mc_star.write("data_general\n\nloop_\n_rlnMotionModelVersion\n0\n\n")
    with open(
        tmp_path / "MotionCorr/job002/Movies/Position_1_002_0.0.star", "w"
    ) as mc_star:
        # Local motion model succeeded
        mc_star.write("data_general\n\nloop_\n_rlnMotionModelVersion\n1\n\n")
        mc_star.write("data_local_motion_model\n\nloop_\n_rlnMotionModelCoeff\n1\n2\n")
    with open(
        tmp_path / "MotionCorr/job002/Movies/Position_1_003_0.0.star", "w"
    ) as mc_star:
        # Local motion model failed
        mc_star.write("data_general\n\nloop_\n_rlnMotionModelVersion\n1\n\n")
        mc_star.write(
            "data_local_motion_model\n\nloop_\n_rlnMotionModelCoeff\n1\n2000\n"
        )

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.st",
        "input_file_list": str(
            [
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_001_0.0.mrc", "1.00"],
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_002_0.0.mrc", "2.00"],
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_003_0.0.mrc", "3.00"],
            ]
        ),
        "pixel_size": 1,
        "relion_options": {},
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["pixel_size"] = 1
    output_relion_options["pixel_size_downscaled"] = 4
    output_relion_options["tomo_size_x"] = 3000
    output_relion_options["tomo_size_y"] = 4000

    # Set up the mock service
    service = tomo_align.TomoAlign(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()

    def write_aretomo_outputs(command, capture_output):
        if command[0] != "AreTomo2":
            return CompletedProcess("", returncode=0)
        # Set up outputs: stack_Imod file like AreTomo2, no exclusions but with space
        (tmp_path / "Tomograms/job006/tomograms/test_stack_Imod").mkdir(parents=True)
        (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc").touch()
        with open(
            tmp_path / "Tomograms/job006/tomograms/test_stack_Imod/tilt.com", "w"
        ) as dark_file:
            dark_file.write("EXCLUDELIST ")
        with open(
            tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w"
        ) as aln_file:
            aln_file.write(
                "dummy 0 1000 1.2 2.3 5 6 7 8 4.5\ndummy 0 1000 1.2 2.3 5 6 7 8 4.5"
            )
        return CompletedProcess(
            "",
            returncode=0,
            stdout="stdout".encode("ascii"),
            stderr="stderr".encode("ascii"),
        )

    mock_subprocess.side_effect = write_aretomo_outputs

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    # Check the expected calls were made
    assert (
        tmp_path / "Tomograms/job006/tomograms/test_stack_xy_shift_plot.json"
    ).is_file()
    assert mock_subprocess.call_count == 3

    # Check the angle file
    assert (
        tmp_path / "Tomograms/job006/tomograms/test_stack_tilt_angles.txt"
    ).is_file()
    with open(
        tmp_path / "Tomograms/job006/tomograms/test_stack_tilt_angles.txt", "r"
    ) as angfile:
        angles_data = angfile.read()
    assert angles_data == "1.00  1\n2.00  2\n"

    # Check the stack file does not have the tilt where motion correction failed
    with open(
        tmp_path / "Tomograms/job006/tomograms/test_stack_newstack.txt", "r"
    ) as newstack_file:
        newstack_tilts = newstack_file.read()
    assert newstack_tilts == (
        f"2\n{tmp_path}/MotionCorr/job002/Movies/Position_1_001_0.0.mrc\n0\n"
        f"{tmp_path}/MotionCorr/job002/Movies/Position_1_002_0.0.mrc\n0\n"
    )

    # Look at a sample of the messages to check they use input files 1 and 2
    assert offline_transport.send.call_count == 14
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.excludetilts",
            "experiment_type": "tomography",
            "input_file": f"{tmp_path}/CtfFind/job003/Movies/Position_1_001_0.0.mrc",
            "output_file": f"{tmp_path}/ExcludeTiltImages/job004/tilts/Position_1_001_0.0.mrc",
            "relion_options": output_relion_options,
            "command": "",
            "stdout": "",
            "stderr": "",
            "success": True,
        },
    )
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.excludetilts",
            "experiment_type": "tomography",
            "input_file": f"{tmp_path}/CtfFind/job003/Movies/Position_1_002_0.0.mrc",
            "output_file": f"{tmp_path}/ExcludeTiltImages/job004/tilts/Position_1_002_0.0.mrc",
            "relion_options": output_relion_options,
            "command": "",
            "stdout": "",
            "stderr": "",
            "success": True,
        },
    )
    offline_transport.send.assert_any_call("success", {})


@mock.patch("cryoemservices.services.tomo_align.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.mrcfile")
def test_tomo_align_service_file_list_rerun(
    mock_mrcfile,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign for a rerun tomogram
    This should call the mock subprocess then send messages on to
    the denoising, ispyb_connector and images services.
    Should not do a node creator send
    """
    mock_mrcfile.open().__enter__().header = {"nx": 3000, "ny": 4000}

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.st",
        "path_pattern": None,
        "input_file_list": str(
            [
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_001_0.0.mrc", "1.00"],
            ]
        ),
        "pixel_size": 1,
        "relion_options": {},
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["pixel_size"] = tomo_align_test_message["pixel_size"]
    output_relion_options["pixel_size_downscaled"] = (
        4 * tomo_align_test_message["pixel_size"]
    )
    output_relion_options["tomo_size_x"] = 3000
    output_relion_options["tomo_size_y"] = 4000
    output_relion_options["tilt_axis_angle"] = 86.0

    # Set up the mock service
    service = tomo_align.TomoAlign(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()

    def write_aretomo_outputs(command, capture_output):
        if command[0] != "AreTomo2":
            return CompletedProcess("", returncode=0)
        # Set up outputs: stack_Imod file like AreTomo2, no exclusions but with space
        with open(
            tmp_path / "Tomograms/job006/tomograms/test_stack_Imod/tilt.com", "w"
        ) as dark_file:
            dark_file.write("EXCLUDELIST ")
        with open(
            tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w"
        ) as aln_file:
            aln_file.write("dummy 86.0 1000 1.2 2.3 5 6 7 8 4.5")
        return CompletedProcess(
            "",
            returncode=0,
            stdout=(
                "Rot center Z 100.0 200.0 3.1\n"
                "Rot center Z 150.0 250.0 2.1\n"
                "Tilt offset 1.1, CC: 0.5\n"
                "Best tilt axis:   57, Score:   0.5\n"
            ).encode("ascii"),
            stderr="stderr".encode("ascii"),
        )

    mock_subprocess.side_effect = write_aretomo_outputs

    # Pre-make the output image so this is a rerun
    (tmp_path / "Tomograms/job006/tomograms/test_stack_Imod").mkdir(parents=True)
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc").touch()

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    # Check the expected calls were made
    assert mock_subprocess.call_count == 3

    # Check the angle file
    assert (
        tmp_path / "Tomograms/job006/tomograms/test_stack_tilt_angles.txt"
    ).is_file()
    with open(
        tmp_path / "Tomograms/job006/tomograms/test_stack_tilt_angles.txt", "r"
    ) as angfile:
        angles_data = angfile.read()
    assert angles_data == "1.00  1\n"

    # Check the shift plot
    with open(
        tmp_path / "Tomograms/job006/tomograms/test_stack_xy_shift_plot.json"
    ) as shift_plot:
        shift_data = json.load(shift_plot)
    assert shift_data["data"][0]["x"] == [1.2]
    assert shift_data["data"][0]["y"] == [2.3]

    # Check that the correct messages were sent - one less node creator
    assert offline_transport.send.call_count == 11
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.excludetilts",
            "experiment_type": "tomography",
            "input_file": f"{tmp_path}/CtfFind/job003/Movies/Position_1_001_0.0.mrc",
            "output_file": f"{tmp_path}/ExcludeTiltImages/job004/tilts/Position_1_001_0.0.mrc",
            "relion_options": output_relion_options,
            "command": "",
            "stdout": "",
            "stderr": "",
            "success": True,
        },
    )
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.aligntiltseries.aretomo",
            "experiment_type": "tomography",
            "input_file": f"{tmp_path}/ExcludeTiltImages/job004/tilts/Position_1_001_0.0.mrc",
            "output_file": f"{tmp_path}/AlignTiltSeries/job005/tilts/Position_1_001_0.0.mrc",
            "relion_options": output_relion_options,
            "command": "",
            "stdout": "",
            "stderr": "",
            "results": {
                "TomoXTilt": "0.00",
                "TomoYTilt": "4.5",
                "TomoZRot": "86.0",
                "TomoXShiftAngst": "1.2",
                "TomoYShiftAngst": "2.3",
            },
            "success": True,
        },
    )
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {"ispyb_command": "multipart_message", "ispyb_command_list": mock.ANY},
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_central_slice",
            "file": tomo_align_test_message["stack_file"],
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_to_apng",
            "file": tomo_align_test_message["stack_file"],
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_central_slice",
            "file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_to_apng",
            "file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_to_jpeg",
            "file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo_projXY.mrc",
            "pixel_spacing": "4.0",
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_to_jpeg",
            "file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo_projXZ.mrc",
            "pixel_spacing": "4.0",
        },
    )
    offline_transport.send.assert_any_call(
        "denoise",
        {
            "volume": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
            "output_dir": f"{tmp_path}/Denoise/job007/tomograms",
            "relion_options": output_relion_options,
        },
    )
    offline_transport.send.assert_any_call("success", {})


@mock.patch("cryoemservices.services.tomo_align.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.mrcfile")
def test_tomo_align_service_path_pattern(
    mock_mrcfile,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign
    This should call the mock subprocess then send messages on to
    the denoising, ispyb_connector and images services.
    """
    mock_mrcfile.open().__enter__().header = {"nx": 3000, "ny": 4000}

    (tmp_path / "MotionCorr/job002/Movies").mkdir(parents=True)
    (tmp_path / "MotionCorr/job002/Movies/Position_1_001_1.00.mrc").touch()
    (tmp_path / "MotionCorr/job002/Movies/Position_1_002_2.00.mrc").touch()

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.st",
        "path_pattern": f"{tmp_path}/MotionCorr/job002/Movies/Position_1_00*.mrc",
        "input_file_list": None,
        "vol_z": 1200,
        "align": 0,
        "out_bin": 4,
        "tilt_axis": 83.0,
        "tilt_cor": 1,
        "flip_int": 1,
        "flip_vol": 1,
        "wbp": 1,
        "roi_file": None,
        "patch": 1,
        "kv": 300,
        "dose_per_frame": 0.2,
        "frame_count": 6,
        "align_file": None,
        "align_z": 500,
        "pixel_size": 1,
        "refine_flag": 1,
        "make_angle_file": True,
        "out_imod": 1,
        "out_imod_xf": None,
        "dark_tol": 0.3,
        "manual_tilt_offset": 10.5,
        "relion_options": {},
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["pixel_size"] = tomo_align_test_message["pixel_size"]
    output_relion_options["pixel_size_downscaled"] = (
        4 * tomo_align_test_message["pixel_size"]
    )
    output_relion_options["tomo_size_x"] = 3000
    output_relion_options["tomo_size_y"] = 4000
    output_relion_options["manual_tilt_offset"] = 10.5
    output_relion_options["frame_count"] = 6
    output_relion_options["dose_per_frame"] = 0.2
    output_relion_options["vol_z"] = 1600

    # Set up the mock service
    service = tomo_align.TomoAlign(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()

    def write_aretomo_outputs(command, capture_output):
        if command[0] != "AreTomo2":
            return CompletedProcess("", returncode=0)
        # Set up outputs: stack_Imod file like AreTomo2, no exclusions without space
        (tmp_path / "Tomograms/job006/tomograms/test_stack_Imod").mkdir(parents=True)
        (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc").touch()
        with open(
            tmp_path / "Tomograms/job006/tomograms/test_stack_Imod/tilt.com", "w"
        ) as dark_file:
            dark_file.write("EXCLUDELIST")
        with open(
            tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w"
        ) as aln_file:
            aln_file.write(
                "dummy 0 1000 1.2 2.3 5 6 7 8 4.5\ndummy 0 1000 1.2 2.3 5 6 7 8 4.5"
            )
        return CompletedProcess(
            "",
            returncode=0,
            stdout="stdout".encode("ascii"),
            stderr="stderr".encode("ascii"),
        )

    mock_subprocess.side_effect = write_aretomo_outputs

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    aretomo_command = [
        "AreTomo2",
        "-InMrc",
        tomo_align_test_message["stack_file"],
        "-OutMrc",
        f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
        "-AngFile",
        f"{tmp_path}/Tomograms/job006/tomograms/test_stack_tilt_angles.txt",
        "-TiltCor",
        "1",
        str(tomo_align_test_message["manual_tilt_offset"]),
        "-VolZ",
        "1600",
        "-TiltAxis",
        str(tomo_align_test_message["tilt_axis"]),
        "1",
        "-ImgDose",
        str(
            tomo_align_test_message["dose_per_frame"]
            * tomo_align_test_message["frame_count"]
        ),
        "-PixSize",
        "1.0",
        "-Align",
        "0",
        "-OutBin",
        str(tomo_align_test_message["out_bin"]),
        "-FlipInt",
        "1",
        "-FlipVol",
        str(tomo_align_test_message["flip_vol"]),
        "-Wbp",
        "1",
        "-Patch",
        "1",
        "-Kv",
        str(tomo_align_test_message["kv"]),
        "-AlignZ",
        str(tomo_align_test_message["align_z"]),
        "-OutImod",
        str(tomo_align_test_message["out_imod"]),
        "-DarkTol",
        str(tomo_align_test_message["dark_tol"]),
    ]

    # Check the expected calls were made
    assert (
        tmp_path / "Tomograms/job006/tomograms/test_stack_xy_shift_plot.json"
    ).is_file()
    assert mock_subprocess.call_count == 2
    mock_subprocess.assert_any_call(
        [
            "newstack",
            "-fileinlist",
            f"{tmp_path}/Tomograms/job006/tomograms/test_stack_newstack.txt",
            "-output",
            tomo_align_test_message["stack_file"],
            "-quiet",
        ],
        capture_output=True,
    )
    mock_subprocess.assert_any_call(
        aretomo_command,
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
    assert angles_data == "1.00  1\n2.00  2\n"

    # No need to check all sent messages
    assert offline_transport.send.call_count == 14
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "experiment_type": "tomography",
            "job_type": "relion.reconstructtomograms",
            "input_file": f"{tmp_path}/MotionCorr/job002/Movies/Position_1_001_1.00.mrc",
            "output_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
            "relion_options": output_relion_options,
            "command": " ".join(aretomo_command),
            "stdout": "stdout",
            "stderr": "stderr",
            "success": True,
        },
    )
    offline_transport.send.assert_any_call("success", {})


@mock.patch("cryoemservices.services.tomo_align.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.mrcfile")
def test_tomo_align_service_dark_images(
    mock_mrcfile,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign for a case with dark images which are removed
    """
    mock_mrcfile.open().__enter__().header = {"nx": 3000, "ny": 4000}

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.st",
        "input_file_list": str(
            [
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_001_0.0.mrc", "0.00"],
                [
                    f"{tmp_path}/MotionCorr/job002/Movies/Position_1_002_0.0.mrc",
                    "12.00",
                ],
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_003_0.0.mrc", "6.00"],
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_004_0.0.mrc", "9.00"],
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_005_0.0.mrc", "3.00"],
            ]
        ),
        "vol_z": 1200,
        "out_bin": 4,
        "tilt_cor": 1,
        "flip_vol": 1,
        "pixel_size": 1,
        "make_angle_file": False,
        "out_imod": 1,
        "relion_options": {},
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["pixel_size"] = tomo_align_test_message["pixel_size"]
    output_relion_options["pixel_size_downscaled"] = (
        4 * tomo_align_test_message["pixel_size"]
    )
    output_relion_options["tomo_size_x"] = 3000
    output_relion_options["tomo_size_y"] = 4000

    # Set up the mock service
    service = tomo_align.TomoAlign(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()

    x_tilts = ["1.2", "2.4", "3.2", "3.4", "4.2"]
    y_tilts = ["2.3", "2.5", "4.3", "4.5", "6.3"]
    tilt_angles = ["0.01", "12.01", "6.01", "9.01", "3.01"]

    def write_aretomo_outputs(command, capture_output):
        if command[0] != "AreTomo2":
            return CompletedProcess("", returncode=0)
        # Set up outputs: stack_aretomo_Imod file like AreTomo, with exclusions and spaces
        (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo_Imod").mkdir(
            parents=True
        )
        (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc").touch()
        with open(
            tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo_Imod/tilt.com",
            "w",
        ) as dark_file:
            dark_file.write("EXCLUDELIST 2, 5")
        with open(
            tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w"
        ) as aln_file:
            for i in [0, 2, 3]:
                aln_file.write(
                    f"dummy 0 1000 {x_tilts[i]} {y_tilts[i]} 5 6 7 8 {tilt_angles[i]}\n"
                )
        return CompletedProcess(
            "",
            returncode=0,
            stdout=(
                "Rot center Z 100.0 200.0 3.1\n"
                "Rot center Z 150.0 250.0 2.1\n"
                "Tilt offset 1.1, CC: 0.5\n"
                "Best tilt axis:   57, Score:   0.5\n"
            ).encode("ascii"),
            stderr="stderr".encode("ascii"),
        )

    mock_subprocess.side_effect = write_aretomo_outputs

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    # Check the aretomo call
    aretomo_command = [
        "AreTomo2",
        "-InMrc",
        tomo_align_test_message["stack_file"],
        "-OutMrc",
        f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
        "-TiltRange",
        "0.00",
        "12.00",
        "-TiltCor",
        "1",
        "-VolZ",
        str(tomo_align_test_message["vol_z"]),
        "-TiltAxis",
        "85",
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
    mock_subprocess.assert_any_call(
        aretomo_command,
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
    assert angles_data == "0.00  1\n3.00  5\n6.00  3\n9.00  4\n12.00  2\n"

    # Check that the correct messages were sent
    assert offline_transport.send.call_count == 16
    # Expect to get messages for three tilts, and not the excluded ones
    for image in [1, 3, 4]:
        offline_transport.send.assert_any_call(
            "node_creator",
            {
                "job_type": "relion.excludetilts",
                "experiment_type": "tomography",
                "input_file": f"{tmp_path}/CtfFind/job003/Movies/Position_1_00{image}_0.0.mrc",
                "output_file": f"{tmp_path}/ExcludeTiltImages/job004/tilts/Position_1_00{image}_0.0.mrc",
                "relion_options": output_relion_options,
                "command": "",
                "stdout": "",
                "stderr": "",
                "success": True,
            },
        )
        offline_transport.send.assert_any_call(
            "node_creator",
            {
                "job_type": "relion.aligntiltseries.aretomo",
                "experiment_type": "tomography",
                "input_file": f"{tmp_path}/ExcludeTiltImages/job004/tilts/Position_1_00{image}_0.0.mrc",
                "output_file": f"{tmp_path}/AlignTiltSeries/job005/tilts/Position_1_00{image}_0.0.mrc",
                "relion_options": output_relion_options,
                "command": "",
                "stdout": "",
                "stderr": "",
                "results": {
                    "TomoXTilt": "0.00",
                    "TomoYTilt": tilt_angles[image - 1],
                    "TomoZRot": "0.0",
                    "TomoXShiftAngst": x_tilts[image - 1],
                    "TomoYShiftAngst": y_tilts[image - 1],
                },
                "success": True,
            },
        )
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "ispyb_command": "multipart_message",
            "ispyb_command_list": [
                {
                    "ispyb_command": "insert_tomogram",
                    "volume_file": "test_stack_aretomo.mrc",
                    "stack_file": tomo_align_test_message["stack_file"],
                    "size_x": 750,
                    "size_y": 1000,
                    "size_z": 300,
                    "pixel_spacing": "4.0",
                    "tilt_angle_offset": "1.1",
                    "z_shift": "2.1",
                    "file_directory": f"{tmp_path}/Tomograms/job006/tomograms",
                    "central_slice_image": "test_stack_aretomo_thumbnail.jpeg",
                    "tomogram_movie": "test_stack_aretomo_movie.png",
                    "xy_shift_plot": "test_stack_xy_shift_plot.json",
                    "proj_xy": "test_stack_aretomo_projXY.jpeg",
                    "proj_xz": "test_stack_aretomo_projXZ.jpeg",
                    "alignment_quality": "0.5",
                },
                {
                    "ispyb_command": "insert_tilt_image_alignment",
                    "psd_file": None,
                    "refined_magnification": "1000.0",
                    "refined_tilt_angle": "0.01",
                    "refined_tilt_axis": "0.0",
                    "path": f"{tmp_path}/MotionCorr/job002/Movies/Position_1_001_0.0.mrc",
                },
                {
                    "ispyb_command": "insert_tilt_image_alignment",
                    "psd_file": None,
                    "refined_magnification": "1000.0",
                    "refined_tilt_angle": "6.01",
                    "refined_tilt_axis": "0.0",
                    "path": f"{tmp_path}/MotionCorr/job002/Movies/Position_1_003_0.0.mrc",
                },
                {
                    "ispyb_command": "insert_tilt_image_alignment",
                    "psd_file": None,
                    "refined_magnification": "1000.0",
                    "refined_tilt_angle": "9.01",
                    "refined_tilt_axis": "0.0",
                    "path": f"{tmp_path}/MotionCorr/job002/Movies/Position_1_004_0.0.mrc",
                },
            ],
        },
    )
    offline_transport.send.assert_any_call("success", {})


@mock.patch("cryoemservices.services.tomo_align.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.mrcfile")
def test_tomo_align_service_all_dark(
    mock_mrcfile,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign for a case where all images are dark
    """
    mock_mrcfile.open().__enter__().header = {"nx": 3000, "ny": 4000}

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.st",
        "input_file_list": str(
            [
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_001_0.0.mrc", "0.00"],
                [
                    f"{tmp_path}/MotionCorr/job002/Movies/Position_1_002_0.0.mrc",
                    "-2.00",
                ],
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_003_0.0.mrc", "2.00"],
                [
                    f"{tmp_path}/MotionCorr/job002/Movies/Position_1_004_0.0.mrc",
                    "-4.00",
                ],
                [f"{tmp_path}/MotionCorr/job002/Movies/Position_1_005_0.0.mrc", "4.00"],
            ]
        ),
        "vol_z": 1200,
        "out_bin": 4,
        "tilt_cor": 1,
        "flip_vol": 1,
        "pixel_size": 1,
        "out_imod": 1,
        "relion_options": {},
    }

    # Set up the mock service
    service = tomo_align.TomoAlign(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()

    def write_aretomo_outputs(command, capture_output):
        if command[0] != "AreTomo2":
            return CompletedProcess("", returncode=0)
        # Set up outputs: stack_Imod file like AreTomo2, with exclusions and no spaces
        (tmp_path / "Tomograms/job006/tomograms/test_stack_Imod").mkdir(parents=True)
        (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc").touch()
        with open(
            tmp_path / "Tomograms/job006/tomograms/test_stack_Imod/tilt.com", "w"
        ) as dark_file:
            dark_file.write("EXCLUDELIST 1,2,3,4,5")
        with open(
            tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w"
        ) as aln_file:
            aln_file.write("")
        return CompletedProcess(
            "",
            returncode=0,
            stdout=(
                "Rot center Z 100.0 200.0 3.1\n"
                "Rot center Z 150.0 250.0 2.1\n"
                "Tilt offset 1.1, CC: 0.5\n"
                "Best tilt axis:   57, Score:   0.5\n"
            ).encode("ascii"),
            stderr="stderr".encode("ascii"),
        )

    mock_subprocess.side_effect = write_aretomo_outputs

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    # Check the angle file
    assert (
        tmp_path / "Tomograms/job006/tomograms/test_stack_tilt_angles.txt"
    ).is_file()
    with open(
        tmp_path / "Tomograms/job006/tomograms/test_stack_tilt_angles.txt", "r"
    ) as angfile:
        angles_data = angfile.read()
    assert angles_data == "-4.00  4\n-2.00  2\n0.00  1\n2.00  3\n4.00  5\n"

    # Check that the correct messages were sent
    assert offline_transport.send.call_count == 10
    # Expect to get messages for three tilts, and not the excluded ones
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "ispyb_command": "multipart_message",
            "ispyb_command_list": [
                {
                    "ispyb_command": "insert_tomogram",
                    "volume_file": "test_stack_aretomo.mrc",
                    "stack_file": tomo_align_test_message["stack_file"],
                    "size_x": 750,
                    "size_y": 1000,
                    "size_z": 300,
                    "pixel_spacing": "4.0",
                    "tilt_angle_offset": "1.1",
                    "z_shift": "2.1",
                    "file_directory": f"{tmp_path}/Tomograms/job006/tomograms",
                    "central_slice_image": "test_stack_aretomo_thumbnail.jpeg",
                    "tomogram_movie": "test_stack_aretomo_movie.png",
                    "xy_shift_plot": "test_stack_xy_shift_plot.json",
                    "proj_xy": "test_stack_aretomo_projXY.jpeg",
                    "proj_xz": "test_stack_aretomo_projXZ.jpeg",
                    "alignment_quality": "0.5",
                },
            ],
        },
    )
    offline_transport.send.assert_any_call("success", {})


def test_parse_tomo_align_output(offline_transport):
    """
    Send test lines to the output parser
    to check the rotations and offsets are being read in
    """
    service = tomo_align.TomoAlign(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()

    tomo_align.TomoAlign.parse_tomo_output(
        service,
        "Rot center Z 100.0 200.0 300.0\n"
        "Rot center Z 150.0 250.0 350.0\n"
        "Tilt offset 1.0, CC: 0.5\n"
        "Best tilt axis:   57, Score:   0.07568",
    )

    assert service.rot_centre_z_list == ["300.0", "350.0"]
    assert service.tilt_offset == 1.0
    assert service.alignment_quality == 0.07568


@mock.patch("cryoemservices.services.tomo_align.subprocess.run")
def test_run_tilt_denoising(mock_subprocess, tmp_path):
    mock_subprocess().returncode = 0

    tilt_in = f"{tmp_path}/processed/relion_murfey/MotionCorr/job002/Movies/tilt.mrc"
    tilt_out = (
        f"{tmp_path}/spool/relion_murfey/MotionCorr/job002/Movies/tilt_denoised.mrc"
    )

    denoised_tilt = tomo_align.run_tilt_denoising(tilt_in)

    assert denoised_tilt == tilt_out

    mock_subprocess.assert_called_with(
        ["python", "run_denoiser.py", "--nimage", tilt_in, "--dimage", tilt_out],
        capture_output=True,
    )
