from __future__ import annotations

import sys
from unittest import mock

import pytest
import zocalo.configuration
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import tomo_align
from cryoemservices.util.relion_service_options import RelionServiceOptions


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
@mock.patch("cryoemservices.services.tomo_align.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.px.scatter")
def test_tomo_align_service(
    mock_plotly, mock_subprocess, mock_environment, offline_transport, tmp_path
):
    """
    Send a test message to TomoAlign
    This should call the mock subprocess then send messages on to
    the denoising, ispyb_connector and images services.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "parameters": {
            "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.st",
            "path_pattern": None,
            "input_file_list": str(
                [[f"{tmp_path}/MotionCorr/job002/Movies/input_file_1.mrc", "1.00"]]
            ),
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

    # Set up the mock service
    service = tomo_align.TomoAlign(environment=mock_environment)
    service.transport = offline_transport
    service.start()

    service.rot_centre_z_list = [1.1, 2.1]
    service.tilt_offset = 1.1
    service.alignment_quality = 0.5

    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo_Imod").mkdir(
        parents=True
    )
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo_Imod/tilt.com").touch()
    with open(tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w") as aln_file:
        aln_file.write("dummy 0 1000 1.2 2.3 5 6 7 8 4.5")

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    aretomo_command = [
        "AreTomo",
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
                "stdout": "",
                "stderr": "",
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
                        "store_result": "ispyb_tomogram_id",
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


def test_parse_tomo_align_output(mock_environment, offline_transport):
    """
    Send test lines to the output parser
    to check the rotations and offsets are being read in
    """
    service = tomo_align.TomoAlign(environment=mock_environment)
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
