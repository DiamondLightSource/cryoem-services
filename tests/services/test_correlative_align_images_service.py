from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.correlative_align_images import (
    AlignImagesParameters,
    AlignImagesService,
)
from cryoemservices.util.models import MockRW


@pytest.fixture
def offline_transport(mocker: MockerFixture):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.parametrize(
    "use_recwrap",
    (
        True,
        False,
    ),
)
def test_align_images_service(
    tmp_path: Path,
    offline_transport: OfflineTransport,
    use_recwrap: bool,
):
    # Set up the message parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    align_images_test_message = {
        "id_ref": 1,
        "image_ref": str(tmp_path / "ref.png"),
        "pixel_size_ref": 1e-6,
        "id_mov": 1,
        "image_mov": str(tmp_path / "mov.png"),
        "pixel_size_mov": 1e-6,
    }
    params = AlignImagesParameters(**align_images_test_message)

    # Set up and run the service
    service = AlignImagesService(environment={"queue": ""}, transport=offline_transport)
    service.log = MagicMock()  # Mock the logger to evaluate calls
    service.initializing()
    if use_recwrap:
        recwrap = MockRW(offline_transport)
        recwrap.recipe_step = {"parameters": align_images_test_message}
        service.call_align_images(
            recwrap,
            header=header,
            message=None,
        )
    else:
        service.call_align_images(
            None,
            header=header,
            message=align_images_test_message,
        )

    # Check that the main block in the function was run
    service.log.info.assert_called_with(
        "Running image alignment with the following parameters:\n"
        f"{params.model_dump(mode='json')}"
    )
