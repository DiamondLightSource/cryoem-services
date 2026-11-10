import json
import subprocess
import uuid
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.sim_atlas import SIMAtlasService, StitchMParameters
from cryoemservices.util.models import MockRW


@pytest.fixture
def offline_transport(mocker: MockerFixture):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "ack")
    return transport


@pytest.mark.parametrize(
    "test_params",
    (  # Use recwrap | Has marker file | Return code | File created
        # Success cases
        (True, True, 0, True),
        (True, False, 0, True),
        (False, True, 0, True),
        (False, False, 0, True),
        (True, True, 1, True),
        (True, True, 0, False),
    ),
)
def test_sim_atlas_service(
    mocker: MockerFixture,
    tmp_path: Path,
    offline_transport: OfflineTransport,
    test_params: tuple[bool, bool, int, bool],
):
    # Unpack test params
    (use_recwrap, has_marker_file, return_code, file_created) = test_params

    # Set up the message parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    visit_name = "cm12345-6"
    visit_dir = tmp_path / visit_name

    # Construct raw files
    raw_dir = visit_dir / "raw" / "some_dir"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mosaic_mrc_file = raw_dir / "mosaic.mrc"
    mosaic_mrc_file.touch(exist_ok=True)
    mosaic_text_file = raw_dir / "mosaic.txt"
    mosaic_text_file.touch(exist_ok=True)
    markers_file = raw_dir / "markers.txt"
    markers_file.touch(exist_ok=True)

    # Mock the UUID creation function
    uid = str(uuid.uuid4())
    mocker.patch("cryoemservices.services.sim_atlas.uuid.uuid4", return_value=uid)

    # Construct file paths to the temporary and final stitched atlas files
    output_dir = visit_dir / "processed" / "some_dir"
    tmp_dir = output_dir / uid
    tmp_dir.mkdir(parents=True, exist_ok=True)

    mosaic_text_file_tmp = tmp_dir / mosaic_text_file.name

    output_file = output_dir / f"{mosaic_mrc_file.stem}.ome.tiff"
    output_file_tmp = tmp_dir / f"{mosaic_mrc_file.stem}.ome.tiff"
    if file_created:
        output_file_tmp.touch(exist_ok=True)

    stitchm_test_message = {
        "visit_name": visit_name,
        "mosaic_mrc_file": str(mosaic_mrc_file),
        "mosaic_text_file": str(mosaic_text_file),
        "markers_file": str(markers_file) if has_marker_file else "None",
        "output_dir": str(output_dir),
    }
    params = StitchMParameters(**stitchm_test_message)

    # Mock the service's reject call
    mock_reject = mocker.patch(
        "cryoemservices.services.common_service.CommonService._reject_message"
    )

    # Construct the subprocess logs
    stdout_lines = "\n".join(
        [
            f"INFO     Running StitchM with arguments: mosaic={mosaic_text_file_tmp}, markers={markers_file}, normalise=True, fl_filter=True",
            f"INFO     Opening mrc file at {mosaic_mrc_file}",
            "INFO     Creating mosaic",
            "INFO     30 potential fluorescence images identified (and 730 brightfield): '378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407' (counted from 0)",
            "INFO     Trimming brightest pixels",
            "INFO     Re-scaling images to uint16",
            "INFO     Creating OME metadata",
            f"INFO     Saving {output_file_tmp}",
        ]
    )
    mock_process_result = MagicMock(returncode=return_code, stdout=stdout_lines)
    mock_run = mocker.patch(
        "cryoemservices.services.sim_atlas.subprocess.run",
        return_value=mock_process_result,
    )

    # Set up and run the service
    service = SIMAtlasService(environment={"queue": ""}, transport=offline_transport)
    service.log = MagicMock()  # Mock the logger to evaluate calls
    service.initializing()
    if use_recwrap:
        recwrap = MockRW(offline_transport)
        recwrap.recipe_step = {"parameters": stitchm_test_message}
    else:
        recwrap = None
    service.call_stitchm(
        recwrap,
        header=header,
        message=stitchm_test_message,
    )

    # Log on validation success
    service.log.info.assert_any_call(
        "Received the following parameters:\n"
        f"{json.dumps(params.model_dump(), indent=2, default=str)}"
    )

    # Subprocess should be called in every case
    cmd = [
        "stitchm",
        "--mosaic",
        str(mosaic_text_file_tmp),
    ]
    if params.markers_file is not None:
        cmd.extend(["--markers", str(params.markers_file)])
    if not params.filter:
        cmd.extend(["--unfiltered"])
    if not params.normalise:
        cmd.extend(["--no-normalisation"])
    mock_run.assert_called_once_with(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Checks to perform if the subprocess was successful
    if file_created and return_code == 0:
        assert output_file.exists()

        murfey_params = {
            "register": "sim.register_atlas",
            "result": {"output_file": str(output_file)},
        }
        service.log.info.assert_any_call(
            "Will submit the following message back to Murfey:\n"
            f"{json.dumps(murfey_params, indent=2, default=str)}"
        )
        service.log.info.assert_called_with("StitchM job completed")
        offline_transport.ack.assert_called_once()
        mock_reject.assert_not_called()

    # Checks to perform if the subprocess call fails
    if return_code or not file_created:
        if return_code:
            service.log.error.assert_called_with(
                f"StitchM subprocess failed with error code {return_code}"
                # Log any stdout as part of the error
                + (
                    f":\n{mock_process_result.stdout}"
                    if mock_process_result.stdout
                    else ""
                )
            )
        if not file_created:
            service.log.error.assert_called_with(
                f"StitchM failed to generate output file for {params.mosaic_mrc_file}"
                # Log any stdout as part of the error
                + (
                    f":\n{mock_process_result.stdout}"
                    if mock_process_result.stdout
                    else ""
                )
            )
        offline_transport.ack.assert_not_called()
        mock_reject.assert_called_once_with(header, transport=mock.ANY, requeue=False)

    # Check that post-subprocess clean-up has occurred
    assert not output_file_tmp.exists() and not tmp_dir.exists()
