import json
from pathlib import Path
from typing import Callable
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.sim_recon import (
    PySIMReconParameters,
    PySIMReconService,
)
from cryoemservices.util.models import MockRW


@pytest.fixture
def offline_transport(mocker: MockerFixture):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.parametrize(
    "test_params",
    (  # Use recwrap | Stringify
        (True, None),
        (True, str),
        (True, json.dumps),
        (False, None),
        (False, str),
        (False, json.dumps),
    ),
)
def test_align_images_service(
    tmp_path: Path,
    offline_transport: OfflineTransport,
    test_params: tuple[bool, Callable | None],
):
    # Unpack test params
    use_recwrap, func = test_params

    # Set up the message parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    blue_params = {
        "wavelength": 452,
        "ls": 0.330,
        "beaddiam": 0.220,
    }
    green_params = {
        "wavelength": 525,
        "ls": 0.394,
    }
    red_params = {
        "wavelength": 605,
        "ls": 0.451,
    }
    far_red_params = {
        "wavelength": 655,
        "ls": 0.521,
    }

    pysimrecon_test_message = {
        "blue_params": func(blue_params) if func else blue_params,
        "green_params": func(green_params) if func else green_params,
        "red_params": func(red_params) if func else red_params,
        "far_red_params": func(far_red_params) if func else far_red_params,
        "output_dir": str(tmp_path / "dummy"),
    }
    params = PySIMReconParameters(**pysimrecon_test_message)

    # Check that the values were parsed correctly
    assert params.blue_params.model_dump(exclude_none=True) == blue_params
    assert params.green_params.model_dump(exclude_none=True) == green_params
    assert params.red_params.model_dump(exclude_none=True) == red_params
    assert params.far_red_params.model_dump(exclude_none=True) == far_red_params

    # Set up and run the service
    service = PySIMReconService(environment={"queue": ""}, transport=offline_transport)
    service.log = MagicMock()  # Mock the logger to evaluate calls
    service.initializing()
    if use_recwrap:
        recwrap = MockRW(offline_transport)
        recwrap.recipe_step = {"parameters": pysimrecon_test_message}
        service.call_pysimrecon(
            recwrap,
            header=header,
            message=None,
        )
    else:
        service.call_pysimrecon(
            None,
            header=header,
            message=pysimrecon_test_message,
        )

    # Check that the main block in the function was run
    service.log.info.assert_called_with(
        "Running PySIMRecon with the following parameters:\n"
        f"{params.model_dump(mode='json')}"
    )
