from subprocess import CompletedProcess
from unittest import mock

import numpy as np
import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import tomo_align_imod


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


class MrcFileHeader:
    def __init__(self, nx, ny, nz=1):
        self.nx = nx
        self.ny = ny
        self.nz = nz


@mock.patch("cryoemservices.services.tomo_align_imod.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align_imod.mrcfile")
@mock.patch("cryoemservices.services.tomo_align_imod.OleFileIO")
@mock.patch("cryoemservices.services.tomo_align_imod.convert_and_save")
def test_tomo_align_imod(
    mock_convert_and_save,
    mock_ole_file,
    mock_mrcfile,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to ImodTomoAlign
    This should call the mock subprocess then send messages on to
    the denoising, ispyb_connector and images services.
    """
    mock_mrcfile.open().__enter__().header = MrcFileHeader(nx=4000, ny=3000, nz=600)
    mock_ole_file().__enter__().exists.return_value = True
    mock_ole_file().__enter__().openstream().getvalue.return_value = np.array(
        [0.01, 0.3, 0.5], dtype=np.float32
    ).tobytes()
    mock_subprocess.return_value = CompletedProcess(
        "",
        returncode=0,
        stdout="stdout".encode("ascii"),
        stderr="stderr".encode("ascii"),
    )

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "stack_file": f"{tmp_path}/recipe/Tomograms/test_stack.mrc",
        "txrm_file": f"{tmp_path}/test.txrm",
        "xrm_reference": f"{tmp_path}/ref.xrm",
        "pixel_size": 106,
        "vol_z": 500,
        "out_bin": 1,
        "cpus": 2,
    }

    # Touch the expected input/output and stack files
    (tmp_path / "test.txrm").touch()
    (tmp_path / "recipe/Tomograms").mkdir(parents=True)
    (tmp_path / "recipe/Tomograms/test_stack.mrc").touch()
    (tmp_path / "recipe/Tomograms/test_stack.rec").touch()

    # Set up the mock service
    service = tomo_align_imod.ImodTomoAlign(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    # Check conversion
    mock_convert_and_save.assert_called_once_with(
        f"{tmp_path}/test.txrm",
        f"{tmp_path}/recipe/Tomograms/test_stack.tiff",
        custom_reference=f"{tmp_path}/ref.xrm",
    )
    mock_subprocess.assert_any_call(
        [
            "tif2mrc",
            f"{tmp_path}/recipe/Tomograms/test_stack.tiff",
            f"{tmp_path}/recipe/Tomograms/test_stack.mrc",
        ],
        capture_output=True,
    )

    # Check txrm file reading
    mock_ole_file.assert_any_call(f"{tmp_path}/test.txrm")
    mock_ole_file().__enter__().exists.assert_any_call("ImageInfo/Angles")
    mock_ole_file().__enter__().openstream.assert_any_call("ImageInfo/Angles")
    assert (tmp_path / "recipe/Tomograms/test_stack.rawtlt").is_file()

    # Check batchruntomo command
    mock_subprocess.assert_any_call(
        [
            "batchruntomo",
            "-directive",
            f"{tmp_path}/recipe/Tomograms/test_stack.adoc",
            "-cpus",
            "2",
        ],
        capture_output=True,
    )

    """# Check the shift plot
    with open(
        tmp_path / "recipe/Tomograms/test_stack.rec/test_stack_xy_shift_plot.json"
    ) as shift_plot:
        shift_data = json.load(shift_plot)
    assert shift_data["data"][0]["x"] == [1.2]
    assert shift_data["data"][0]["y"] == [2.3]
    """
    # Check that the correct messages were sent
    assert offline_transport.send.call_count == 9  # 10
    """
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "tilt_series_alignment",
            "file": tomo_align_test_message["stack_file"],
            "aln_file": "dummy",  # f"{tmp_path}/Tomograms/job006/tomograms/test_stack.aln",
            "pixel_size": tomo_align_test_message["pixel_size"],
        },
    )
    """
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
            "file": f"{tmp_path}/recipe/Tomograms/test_stack.rec",
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_to_apng",
            "file": f"{tmp_path}/recipe/Tomograms/test_stack.rec",
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_projection",
            "file": f"{tmp_path}/recipe/Tomograms/test_stack.rec",
            "projection": "XY",
            "pixel_spacing": 106,
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_projection",
            "file": f"{tmp_path}/recipe/Tomograms/test_stack.rec",
            "projection": "YZ",
            "pixel_spacing": 106,
            "thickness_ang": 500 * 106,
        },
    )
    offline_transport.send.assert_any_call(
        "denoise",
        {
            "volume": f"{tmp_path}/recipe/Tomograms/test_stack.rec",
            "output_dir": f"{tmp_path}/recipe/Denoise",
            "relion_options": {},
        },
    )
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "ispyb_command": "multipart_message",
            "ispyb_command_list": [
                {
                    "ispyb_command": "insert_tomogram",
                    "volume_file": "test_stack.rec",
                    "stack_file": tomo_align_test_message["stack_file"],
                    "size_x": 4000,
                    "size_y": 3000,
                    "size_z": 500,
                    "pixel_spacing": 106.0,
                    "z_shift": 0,
                    "file_directory": f"{tmp_path}/recipe/Tomograms",
                    "central_slice_image": "test_stack_thumbnail.jpeg",
                    "tomogram_movie": "test_stack_movie.png",
                    "xy_shift_plot": "test_stack_xy_shift_plot.json",
                    "proj_xy": "test_stack_projXY.jpeg",
                    "proj_xz": "test_stack_projYZ.jpeg",
                },
                {
                    "ispyb_command": "insert_processed_tomogram",
                    "file_path": tomo_align_test_message["stack_file"],
                    "processing_type": "Stack",
                },
                {
                    "ispyb_command": "insert_processed_tomogram",
                    "file_path": f"{tmp_path}/recipe/Tomograms/test_stack_alignment.jpeg",
                    "processing_type": "Alignment",
                },
            ],
        },
    )
    offline_transport.send.assert_any_call("success", {})


def test_write_batch_directive_patch_wbp(tmp_path):
    returned_adoc = tomo_align_imod.write_batch_directive_file(
        tomo_align_imod.ImodTomoParameters(
            **{
                "stack_file": f"{tmp_path}/stack.mrc",
                "txrm_file": f"{tmp_path}/stack.txrm",
                "pixel_size": 10.5,
                "patch": 1,
                "patch_size": 2,
                "patch_overlap": 0.7,
                "vol_z": 500,
                "wbp": 1,
                "sirt": 0,
                "sirt_leave_iterations": 10,
            }
        )
    )
    assert returned_adoc == tmp_path / "stack.adoc"
    with open(returned_adoc) as f:
        adoc_lines = f.readlines()

    # Check some of the basic lines
    assert f"setupset.datasetDirectory={tmp_path}\n" in adoc_lines
    assert "setupset.copyarg.name=stack\n" in adoc_lines
    assert "setupset.copyarg.pixel=1.05\n" in adoc_lines
    assert "comparam.tilt.tilt.THICKNESS=500\n" in adoc_lines
    assert "runtime.Reconstruction.any.useSirt=0\n" in adoc_lines
    assert "runtime.Reconstruction.any.doBackprojAlso=1\n" in adoc_lines

    # Check patch lines
    assert "comparam.xcorr_pt.tiltxcorr.SizeOfPatchesXandY=2 2\n" in adoc_lines
    assert "comparam.xcorr_pt.tiltxcorr.OverlapOfPatchesXandY=0.7 0.7\n" in adoc_lines

    # Check lines not present
    assert "comparam.autofidseed.autofidseed.AdjustSizes=1\n" not in adoc_lines
    assert "comparam.sirtsetup.sirtsetup.LeaveIterations=10\n" not in adoc_lines


def test_write_batch_directive_beads_sirt(tmp_path):
    returned_adoc = tomo_align_imod.write_batch_directive_file(
        tomo_align_imod.ImodTomoParameters(
            **{
                "stack_file": f"{tmp_path}/stack.mrc",
                "txrm_file": f"{tmp_path}/stack.txrm",
                "bead_count": 7,
                "pixel_size": 10.5,
                "patch": 0,
                "patch_size": 2,
                "patch_overlap": 4,
                "vol_z": 500,
                "wbp": 0,
                "sirt": 1,
                "sirt_leave_iterations": 10,
            }
        )
    )
    assert returned_adoc == tmp_path / "stack.adoc"
    with open(returned_adoc) as f:
        adoc_lines = f.readlines()

    # Check some of the basic lines
    assert f"setupset.datasetDirectory={tmp_path}\n" in adoc_lines
    assert "setupset.copyarg.name=stack\n" in adoc_lines
    assert "setupset.copyarg.pixel=1.05\n" in adoc_lines
    assert "comparam.tilt.tilt.THICKNESS=500\n" in adoc_lines
    assert "runtime.Reconstruction.any.useSirt=1\n" in adoc_lines
    assert "runtime.Reconstruction.any.doBackprojAlso=0\n" in adoc_lines

    # Check bead and sirt lines
    assert "comparam.track.beadtrack.LightBeads=0\n" in adoc_lines
    assert "comparam.autofidseed.autofidseed.TargetNumberOfBeads=7\n" in adoc_lines
    assert "comparam.sirtsetup.sirtsetup.LeaveIterations=10\n" in adoc_lines

    # Check lines not present
    assert "comparam.xcorr_pt.tiltxcorr.IterateCorrelations=1\n" not in adoc_lines
    assert "comparam.xcorr_pt.tiltxcorr.OverlapOfPatchesXandY=4 4\n" not in adoc_lines
