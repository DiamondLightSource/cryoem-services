import json
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
    (  # Use recwrap | Stringify
        # Success cases
        (True,),
        (False,),
    ),
)
def test_sim_atlas_service(
    tmp_path: Path,
    offline_transport: OfflineTransport,
    test_params: tuple[bool],
):
    # Unpack test params
    (use_recwrap,) = test_params

    # Set up the message parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    visit_name = "cm12345-6"
    visit_dir = tmp_path / visit_name
    mosaic_mrc_file = visit_dir / "raw" / "some_dir" / "test_file"
    mosaic_text_file = visit_dir / "raw" / "some_dir" / "test_file"
    markers_file = visit_dir / "raw" / "some_dir" / "test_file"
    output_dir = tmp_path / "processed" / "some_dir"

    stitchm_test_message = {
        "visit_name": visit_name,
        "mosaic_mrc_file": str(mosaic_mrc_file),
        "mosaic_text_file": str(mosaic_text_file),
        "markers_file": str(markers_file),
        "output_dir": str(output_dir),
    }
    params = StitchMParameters(**stitchm_test_message)

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

    # Check that the message was acked
    service.log.info.assert_called_with("StitchM job completed")
    offline_transport.ack.assert_called_once()
