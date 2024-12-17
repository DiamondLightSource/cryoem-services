from __future__ import annotations

import sys
import time
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


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.tomo_align.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.px.scatter")
@mock.patch("cryoemservices.services.tomo_align.mrcfile")
def test_tomo_align_service_file_list(
    mock_mrcfile,
    mock_plotly,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign
    This should call the mock subprocess then send messages on to
    the denoising, ispyb_connector and images services.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    mock_mrcfile.open().__enter__().header = {"nx": 3000, "ny": 4000}

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "parameters": {
            "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.st",
            "path_pattern": None,
            "input_file_list": str(
                [
                    [f"{tmp_path}/MotionCorr/job002/Movies/input_file_1.mrc", "1.00"],
                ]
            ),
            "vol_z": 1200,
            "align": None,
            "out_bin": 4,
            "tilt_axis": None,
            "tilt_cor": 1,
            "flip_int": None,
            "flip_vol": 1,
            "wbp": None,
            "roi_file": None,
            "patch": None,
            "kv": None,
            "align_file": None,
            "angle_file": f"{tmp_path}/angles.file",
            "align_z": None,
            "pixel_size": 1e-10,
            "init_val": None,
            "refine_flag": None,
            "out_imod": 1,
            "out_imod_xf": None,
            "dark_tol": None,
            "manual_tilt_offset": None,
            "relion_options": {},
        },
        "content": "dummy",
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["pixel_size"] = tomo_align_test_message["parameters"][
        "pixel_size"
    ]
    output_relion_options["pixel_size_downscaled"] = (
        4 * tomo_align_test_message["parameters"]["pixel_size"]
    )
    output_relion_options["tomo_size_x"] = 4000
    output_relion_options["tomo_size_y"] = 3000

    # Set up the mock service
    service = tomo_align.TomoAlign()
    service.transport = offline_transport
    service.start()

    service.rot_centre_z_list = [1.1, 2.1]
    service.tilt_offset = 1.1
    service.alignment_quality = 0.5

    # Set up outputs: stack_Imod file like AreTomo2, no exclusions but with space
    (tmp_path / "Tomograms/job006/tomograms/test_stack_Imod").mkdir(parents=True)
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc").touch()
    with open(
        tmp_path / "Tomograms/job006/tomograms/test_stack_Imod/tilt.com", "w"
    ) as dark_file:
        dark_file.write("EXCLUDELIST ")
    with open(tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w") as aln_file:
        aln_file.write("dummy 0 1000 1.2 2.3 5 6 7 8 4.5")

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    aretomo_command = [
        "AreTomo2",
        "-OutMrc",
        f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
        "-AngFile",
        f"{tmp_path}/angles.file",
        "-TiltCor",
        "1",
        "-InMrc",
        tomo_align_test_message["parameters"]["stack_file"],
        "-PixSize",
        "1e-10",
        "-VolZ",
        str(tomo_align_test_message["parameters"]["vol_z"]),
        "-OutBin",
        str(tomo_align_test_message["parameters"]["out_bin"]),
        "-FlipVol",
        str(tomo_align_test_message["parameters"]["flip_vol"]),
        "-OutImod",
        str(tomo_align_test_message["parameters"]["out_imod"]),
    ]

    # Check the expected calls were made
    assert mock_plotly.call_count == 1
    assert mock_subprocess.call_count == 5
    mock_subprocess.assert_any_call(
        [
            "newstack",
            "-fileinlist",
            f"{tmp_path}/Tomograms/job006/tomograms/test_stack_newstack.txt",
            "-output",
            tomo_align_test_message["parameters"]["stack_file"],
            "-quiet",
        ]
    )
    mock_subprocess.assert_any_call(
        aretomo_command,
        capture_output=True,
    )

    # Check that the correct messages were sent
    assert offline_transport.send.call_count == 10
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "job_type": "relion.excludetilts",
                "experiment_type": "tomography",
                "input_file": f"{tmp_path}/MotionCorr/job002/Movies/input_file_1.mrc",
                "output_file": f"{tmp_path}/ExcludeTiltImages/job004/tilts/input_file_1.mrc",
                "relion_options": output_relion_options,
                "command": "",
                "stdout": "",
                "stderr": "",
                "success": True,
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "job_type": "relion.aligntiltseries",
                "experiment_type": "tomography",
                "input_file": f"{tmp_path}/MotionCorr/job002/Movies/input_file_1.mrc",
                "output_file": f"{tmp_path}/AlignTiltSeries/job005/tilts/input_file_1.mrc",
                "relion_options": output_relion_options,
                "command": "",
                "stdout": "",
                "stderr": "",
                "results": {
                    "TomoXTilt": "0.00",
                    "TomoYTilt": "4.5",
                    "TomoZRot": "0.0",
                    "TomoXShiftAngst": "1.2",
                    "TomoYShiftAngst": "2.3",
                },
                "success": True,
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "experiment_type": "tomography",
                "job_type": "relion.reconstructtomograms",
                "input_file": f"{tmp_path}/MotionCorr/job002/Movies/input_file_1.mrc",
                "output_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
                "relion_options": output_relion_options,
                "command": " ".join(aretomo_command),
                "stdout": "stdout",
                "stderr": "stderr",
                "success": True,
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="ispyb_connector",
        message={
            "parameters": {
                "ispyb_command": "multipart_message",
                "ispyb_command_list": [
                    {
                        "ispyb_command": "insert_tomogram",
                        "volume_file": "test_stack_aretomo.mrc",
                        "stack_file": tomo_align_test_message["parameters"][
                            "stack_file"
                        ],
                        "size_x": None,
                        "size_y": None,
                        "size_z": None,
                        "pixel_spacing": "4e-10",
                        "tilt_angle_offset": "1.1",
                        "z_shift": 2.1,
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
                        "refined_tilt_axis": "0.0",
                        "path": f"{tmp_path}/MotionCorr/job002/Movies/input_file_1.mrc",
                    },
                ],
            },
            "content": {"dummy": "dummy"},
        },
    )
    offline_transport.send.assert_any_call(
        destination="images",
        message={
            "image_command": "mrc_central_slice",
            "file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
        },
    )
    offline_transport.send.assert_any_call(
        destination="images",
        message={
            "image_command": "mrc_to_apng",
            "file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
        },
    )
    offline_transport.send.assert_any_call(
        destination="images",
        message={
            "image_command": "mrc_to_jpeg",
            "file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo_projXY.mrc",
        },
    )
    offline_transport.send.assert_any_call(
        destination="images",
        message={
            "image_command": "mrc_to_jpeg",
            "file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo_projXZ.mrc",
        },
    )
    offline_transport.send.assert_any_call(
        destination="denoise",
        message={
            "volume": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
            "output_dir": f"{tmp_path}/Denoise/job007/tomograms",
            "relion_options": output_relion_options,
        },
    )
    offline_transport.send.assert_any_call(destination="success", message="")


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.tomo_align.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.px.scatter")
@mock.patch("cryoemservices.services.tomo_align.mrcfile")
def test_tomo_align_service_file_list_repeated_tilt(
    mock_mrcfile,
    mock_plotly,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign with a duplicated tilt angle
    Only the newest one of the duplicated tilts should be used
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    mock_mrcfile.open().__enter__().header = {"nx": 3000, "ny": 4000}

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "parameters": {
            "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.st",
            "input_file_list": str(
                [
                    [f"{tmp_path}/MotionCorr/job002/Movies/input_file_1.mrc", "1.00"],
                    [f"{tmp_path}/MotionCorr/job002/Movies/input_file_2.mrc", "1.00"],
                    [f"{tmp_path}/MotionCorr/job002/Movies/input_file_3.mrc", "1.00"],
                ]
            ),
            "pixel_size": 1e-10,
            "relion_options": {},
        },
        "content": "dummy",
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["pixel_size"] = 1e-10
    output_relion_options["pixel_size_downscaled"] = 4e-10
    output_relion_options["tomo_size_x"] = 4000
    output_relion_options["tomo_size_y"] = 3000

    # Create the input files. Needs sleeps to ensure distinct timestamps
    (tmp_path / "MotionCorr/job002/Movies").mkdir(parents=True)
    (tmp_path / "MotionCorr/job002/Movies/input_file_1.mrc").touch()
    time.sleep(1)
    (tmp_path / "MotionCorr/job002/Movies/input_file_2.mrc").touch()
    time.sleep(1)
    (tmp_path / "MotionCorr/job002/Movies/input_file_3.mrc").touch()

    # Set up the mock service
    service = tomo_align.TomoAlign()
    service.transport = offline_transport
    service.start()

    service.rot_centre_z_list = [1.1, 2.1]
    service.tilt_offset = 1.1
    service.alignment_quality = 0.5

    # Set up outputs: stack_Imod file like AreTomo2, no exclusions but with space
    (tmp_path / "Tomograms/job006/tomograms/test_stack_Imod").mkdir(parents=True)
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc").touch()
    with open(
        tmp_path / "Tomograms/job006/tomograms/test_stack_Imod/tilt.com", "w"
    ) as dark_file:
        dark_file.write("EXCLUDELIST ")
    with open(tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w") as aln_file:
        aln_file.write("dummy 0 1000 1.2 2.3 5 6 7 8 4.5")

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    # Check the expected calls were made
    assert mock_plotly.call_count == 1
    assert mock_subprocess.call_count == 5

    # Check the stack file has only the last one of the duplicated tilt angles
    with open(
        tmp_path / "Tomograms/job006/tomograms/test_stack_newstack.txt", "r"
    ) as newstack_file:
        newstack_tilts = newstack_file.read()
    assert (
        newstack_tilts
        == f"1\n{tmp_path}/MotionCorr/job002/Movies/input_file_3.mrc\n0\n"
    )

    # Look at a sample of the messages to check they use input_file_3
    assert offline_transport.send.call_count == 10
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "job_type": "relion.excludetilts",
                "experiment_type": "tomography",
                "input_file": f"{tmp_path}/MotionCorr/job002/Movies/input_file_3.mrc",
                "output_file": f"{tmp_path}/ExcludeTiltImages/job004/tilts/input_file_3.mrc",
                "relion_options": output_relion_options,
                "command": "",
                "stdout": "",
                "stderr": "",
                "success": True,
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(destination="success", message="")


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.tomo_align.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.px.scatter")
@mock.patch("cryoemservices.services.tomo_align.mrcfile")
def test_tomo_align_service_path_pattern(
    mock_mrcfile,
    mock_plotly,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign
    This should call the mock subprocess then send messages on to
    the denoising, ispyb_connector and images services.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    mock_mrcfile.open().__enter__().header = {"nx": 3000, "ny": 4000}

    (tmp_path / "MotionCorr/job002/Movies").mkdir(parents=True)
    (tmp_path / "MotionCorr/job002/Movies/input_file_1.00.mrc").touch()
    (tmp_path / "MotionCorr/job002/Movies/input_file_2.00.mrc").touch()

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "parameters": {
            "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.st",
            "path_pattern": f"{tmp_path}/MotionCorr/job002/Movies/input_file_*.mrc",
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
            "align_file": None,
            "angle_file": f"{tmp_path}/angles.file",
            "align_z": 500,
            "pixel_size": 1e-10,
            "init_val": None,
            "refine_flag": None,
            "out_imod": 1,
            "out_imod_xf": None,
            "dark_tol": 0.3,
            "manual_tilt_offset": 10.5,
            "relion_options": {},
        },
        "content": "dummy",
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["pixel_size"] = tomo_align_test_message["parameters"][
        "pixel_size"
    ]
    output_relion_options["pixel_size_downscaled"] = (
        4 * tomo_align_test_message["parameters"]["pixel_size"]
    )
    output_relion_options["tomo_size_x"] = 4000
    output_relion_options["tomo_size_y"] = 3000
    output_relion_options["manual_tilt_offset"] = 10.5

    # Set up the mock service
    service = tomo_align.TomoAlign()
    service.transport = offline_transport
    service.start()

    service.rot_centre_z_list = [1.1, 2.1]
    service.tilt_offset = 1.1
    service.alignment_quality = 0.5

    # Set up outputs: stack_Imod file like AreTomo2, no exclusions without space
    (tmp_path / "Tomograms/job006/tomograms/test_stack_Imod").mkdir(parents=True)
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc").touch()
    with open(
        tmp_path / "Tomograms/job006/tomograms/test_stack_Imod/tilt.com", "w"
    ) as dark_file:
        dark_file.write("EXCLUDELIST")
    with open(tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w") as aln_file:
        aln_file.write(
            "dummy 0 1000 1.2 2.3 5 6 7 8 4.5\ndummy 0 1000 1.2 2.3 5 6 7 8 4.5"
        )

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    aretomo_command = [
        "AreTomo2",
        "-OutMrc",
        f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
        "-AngFile",
        f"{tmp_path}/angles.file",
        "-TiltCor",
        "1",
        str(tomo_align_test_message["parameters"]["manual_tilt_offset"]),
        "-InMrc",
        tomo_align_test_message["parameters"]["stack_file"],
        "-PixSize",
        "1e-10",
        "-VolZ",
        str(tomo_align_test_message["parameters"]["vol_z"]),
        "-Align",
        "0",
        "-OutBin",
        str(tomo_align_test_message["parameters"]["out_bin"]),
        "-TiltAxis",
        str(tomo_align_test_message["parameters"]["tilt_axis"]),
        "-FlipInt",
        "1",
        "-FlipVol",
        str(tomo_align_test_message["parameters"]["flip_vol"]),
        "-Wbp",
        "1",
        "-Patch",
        "1",
        "-Kv",
        str(tomo_align_test_message["parameters"]["kv"]),
        "-AlignZ",
        str(tomo_align_test_message["parameters"]["align_z"]),
        "-OutImod",
        str(tomo_align_test_message["parameters"]["out_imod"]),
        "-DarkTol",
        str(tomo_align_test_message["parameters"]["dark_tol"]),
    ]

    # Check the expected calls were made
    assert mock_plotly.call_count == 1
    assert mock_subprocess.call_count == 5
    mock_subprocess.assert_any_call(
        [
            "newstack",
            "-fileinlist",
            f"{tmp_path}/Tomograms/job006/tomograms/test_stack_newstack.txt",
            "-output",
            tomo_align_test_message["parameters"]["stack_file"],
            "-quiet",
        ]
    )
    mock_subprocess.assert_any_call(
        aretomo_command,
        capture_output=True,
    )

    # No need to check all sent messages
    assert offline_transport.send.call_count == 12
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "experiment_type": "tomography",
                "job_type": "relion.reconstructtomograms",
                "input_file": f"{tmp_path}/MotionCorr/job002/Movies/input_file_1.00.mrc",
                "output_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc",
                "relion_options": output_relion_options,
                "command": " ".join(aretomo_command),
                "stdout": "stdout",
                "stderr": "stderr",
                "success": True,
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(destination="success", message="")


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.tomo_align.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.px.scatter")
@mock.patch("cryoemservices.services.tomo_align.mrcfile")
def test_tomo_align_service_dark_images(
    mock_mrcfile,
    mock_plotly,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign for a case with dark images which are removed
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    mock_mrcfile.open().__enter__().header = {"nx": 3000, "ny": 4000}

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "parameters": {
            "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.st",
            "input_file_list": str(
                [
                    [f"{tmp_path}/MotionCorr/job002/Movies/input_file_1.mrc", "0.00"],
                    [f"{tmp_path}/MotionCorr/job002/Movies/input_file_2.mrc", "2.00"],
                    [f"{tmp_path}/MotionCorr/job002/Movies/input_file_3.mrc", "4.00"],
                    [f"{tmp_path}/MotionCorr/job002/Movies/input_file_4.mrc", "6.00"],
                    [f"{tmp_path}/MotionCorr/job002/Movies/input_file_5.mrc", "8.00"],
                ]
            ),
            "vol_z": 1200,
            "out_bin": 4,
            "tilt_cor": 1,
            "flip_vol": 1,
            "angle_file": f"{tmp_path}/angles.file",
            "pixel_size": 1e-10,
            "out_imod": 1,
            "relion_options": {},
        },
        "content": "dummy",
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["pixel_size"] = tomo_align_test_message["parameters"][
        "pixel_size"
    ]
    output_relion_options["pixel_size_downscaled"] = (
        4 * tomo_align_test_message["parameters"]["pixel_size"]
    )
    output_relion_options["tomo_size_x"] = 4000
    output_relion_options["tomo_size_y"] = 3000

    # Set up the mock service
    service = tomo_align.TomoAlign()
    service.transport = offline_transport
    service.start()

    service.rot_centre_z_list = [1.1, 2.1]
    service.tilt_offset = 1.1
    service.alignment_quality = 0.5

    x_tilts = ["1.2", "2.4", "3.2", "3.4", "4.2"]
    y_tilts = ["2.3", "2.5", "4.3", "4.5", "6.3"]
    tilt_angles = ["0.01", "2.01", "4.01", "6.01", "8.01"]

    # Set up outputs: stack_aretomo_Imod file like AreTomo, with exclusions and spaces
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo_Imod").mkdir(
        parents=True
    )
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc").touch()
    with open(
        tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo_Imod/tilt.com", "w"
    ) as dark_file:
        dark_file.write("EXCLUDELIST 2, 5")
    with open(tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w") as aln_file:
        for i in [0, 2, 3]:
            aln_file.write(
                f"dummy 0 1000 {x_tilts[i]} {y_tilts[i]} 5 6 7 8 {tilt_angles[i]}\n"
            )

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    # Check that the correct messages were sent
    assert offline_transport.send.call_count == 14
    # Expect to get messages for three tilts, and not the excluded ones
    for image in [1, 3, 4]:
        offline_transport.send.assert_any_call(
            destination="node_creator",
            message={
                "parameters": {
                    "job_type": "relion.excludetilts",
                    "experiment_type": "tomography",
                    "input_file": f"{tmp_path}/MotionCorr/job002/Movies/input_file_{image}.mrc",
                    "output_file": f"{tmp_path}/ExcludeTiltImages/job004/tilts/input_file_{image}.mrc",
                    "relion_options": output_relion_options,
                    "command": "",
                    "stdout": "",
                    "stderr": "",
                    "success": True,
                },
                "content": "dummy",
            },
        )
        offline_transport.send.assert_any_call(
            destination="node_creator",
            message={
                "parameters": {
                    "job_type": "relion.aligntiltseries",
                    "experiment_type": "tomography",
                    "input_file": f"{tmp_path}/MotionCorr/job002/Movies/input_file_{image}.mrc",
                    "output_file": f"{tmp_path}/AlignTiltSeries/job005/tilts/input_file_{image}.mrc",
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
                "content": "dummy",
            },
        )
    offline_transport.send.assert_any_call(
        destination="ispyb_connector",
        message={
            "parameters": {
                "ispyb_command": "multipart_message",
                "ispyb_command_list": [
                    {
                        "ispyb_command": "insert_tomogram",
                        "volume_file": "test_stack_aretomo.mrc",
                        "stack_file": tomo_align_test_message["parameters"][
                            "stack_file"
                        ],
                        "size_x": None,
                        "size_y": None,
                        "size_z": None,
                        "pixel_spacing": "4e-10",
                        "tilt_angle_offset": "1.1",
                        "z_shift": 2.1,
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
                        "path": f"{tmp_path}/MotionCorr/job002/Movies/input_file_1.mrc",
                    },
                    {
                        "ispyb_command": "insert_tilt_image_alignment",
                        "psd_file": None,
                        "refined_magnification": "1000.0",
                        "refined_tilt_angle": "4.01",
                        "refined_tilt_axis": "0.0",
                        "path": f"{tmp_path}/MotionCorr/job002/Movies/input_file_3.mrc",
                    },
                    {
                        "ispyb_command": "insert_tilt_image_alignment",
                        "psd_file": None,
                        "refined_magnification": "1000.0",
                        "refined_tilt_angle": "6.01",
                        "refined_tilt_axis": "0.0",
                        "path": f"{tmp_path}/MotionCorr/job002/Movies/input_file_4.mrc",
                    },
                ],
            },
            "content": {"dummy": "dummy"},
        },
    )
    offline_transport.send.assert_any_call(destination="success", message="")


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.tomo_align.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.px.scatter")
@mock.patch("cryoemservices.services.tomo_align.mrcfile")
def test_tomo_align_service_all_dark(
    mock_mrcfile,
    mock_plotly,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign for a case where all images are dark
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    mock_mrcfile.open().__enter__().header = {"nx": 3000, "ny": 4000}

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "parameters": {
            "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.st",
            "input_file_list": str(
                [
                    [f"{tmp_path}/MotionCorr/job002/Movies/input_file_1.mrc", "0.00"],
                    [f"{tmp_path}/MotionCorr/job002/Movies/input_file_2.mrc", "2.00"],
                    [f"{tmp_path}/MotionCorr/job002/Movies/input_file_3.mrc", "4.00"],
                    [f"{tmp_path}/MotionCorr/job002/Movies/input_file_4.mrc", "6.00"],
                    [f"{tmp_path}/MotionCorr/job002/Movies/input_file_5.mrc", "8.00"],
                ]
            ),
            "vol_z": 1200,
            "out_bin": 4,
            "tilt_cor": 1,
            "flip_vol": 1,
            "angle_file": f"{tmp_path}/angles.file",
            "pixel_size": 1e-10,
            "out_imod": 1,
            "relion_options": {},
        },
        "content": "dummy",
    }

    # Set up the mock service
    service = tomo_align.TomoAlign()
    service.transport = offline_transport
    service.start()

    service.rot_centre_z_list = [1.1, 2.1]
    service.tilt_offset = 1.1
    service.alignment_quality = 0.5

    # Set up outputs: stack_Imod file like AreTomo2, with exclusions and no spaces
    (tmp_path / "Tomograms/job006/tomograms/test_stack_Imod").mkdir(parents=True)
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc").touch()
    with open(
        tmp_path / "Tomograms/job006/tomograms/test_stack_Imod/tilt.com", "w"
    ) as dark_file:
        dark_file.write("EXCLUDELIST 1,2,3,4,5")
    with open(tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w") as aln_file:
        aln_file.write("")

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    # Check that the correct messages were sent
    assert offline_transport.send.call_count == 8
    # Expect to get messages for three tilts, and not the excluded ones
    offline_transport.send.assert_any_call(
        destination="ispyb_connector",
        message={
            "parameters": {
                "ispyb_command": "multipart_message",
                "ispyb_command_list": [
                    {
                        "ispyb_command": "insert_tomogram",
                        "volume_file": "test_stack_aretomo.mrc",
                        "stack_file": tomo_align_test_message["parameters"][
                            "stack_file"
                        ],
                        "size_x": None,
                        "size_y": None,
                        "size_z": None,
                        "pixel_spacing": "4e-10",
                        "tilt_angle_offset": "1.1",
                        "z_shift": 2.1,
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
            "content": {"dummy": "dummy"},
        },
    )
    offline_transport.send.assert_any_call(destination="success", message="")


def test_parse_tomo_align_output(offline_transport):
    """
    Send test lines to the output parser
    to check the rotations and offsets are being read in
    """
    service = tomo_align.TomoAlign()
    service.transport = offline_transport
    service.start()

    tomo_align.TomoAlign.parse_tomo_output(service, "Rot center Z 100.0 200.0 300.0")
    tomo_align.TomoAlign.parse_tomo_output(service, "Rot center Z 150.0 250.0 350.0")
    tomo_align.TomoAlign.parse_tomo_output(service, "Tilt offset 1.0, CC: 0.5")
    tomo_align.TomoAlign.parse_tomo_output(
        service, "Best tilt axis:   57, Score:   0.07568"
    )

    assert service.rot_centre_z_list == ["300.0", "350.0"]
    assert service.tilt_offset == 1.0
    assert service.alignment_quality == 0.07568
