import json
from pathlib import Path
from typing import Callable, cast
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.sim_recon import (
    PySIMReconParameters,
    SIMOTFParameters,
    SIMReconParameters,
    SIMReconService,
    WavelengthParameters,
)
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
        (True, None),
        (True, str),
        (True, json.dumps),
        (False, None),
        (False, str),
        (False, json.dumps),
    ),
)
def test_sim_recon_service(
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
    visit_name = "cm12345-6"
    visit_dir = tmp_path / visit_name
    test_file = visit_dir / "raw" / "some_dir" / "test_file"
    output_dir = tmp_path / "processed" / "some_dir"
    blue_params = {
        "wavelength": 452,
        "ls": 0.330,
        "beaddiam": 0.220,
        "otf_path": str(visit_dir / "setup" / "OTFs" / "452.tiff"),
    }
    green_params = {
        "wavelength": 525,
        "ls": 0.394,
        "otf_path": str(visit_dir / "setup" / "OTFs" / "525.tiff"),
    }
    red_params = {
        "wavelength": 605,
        "ls": 0.451,
        "otf_path": str(visit_dir / "setup" / "OTFs" / "605.tiff"),
    }
    far_red_params = {
        "wavelength": 655,
        "ls": 0.521,
        "otf_path": str(visit_dir / "setup" / "OTFs" / "655.tiff"),
    }

    pysimrecon_test_message = {
        "visit_name": visit_name,
        "file": str(test_file),
        "output_dir": str(output_dir),
        "blue_params": func(blue_params) if func else blue_params,
        "green_params": func(green_params) if func else green_params,
        "red_params": func(red_params) if func else red_params,
        "far_red_params": func(far_red_params) if func else far_red_params,
    }
    params = PySIMReconParameters(**pysimrecon_test_message)

    # Check that the values were parsed correctly
    assert params.visit_name == visit_name
    assert params.file == test_file
    assert params.output_dir == output_dir
    for field_name, wavelength_dict in (
        ("blue_params", blue_params),
        ("green_params", green_params),
        ("red_params", red_params),
        ("far_red_params", far_red_params),
    ):
        wavelength_params = cast(WavelengthParameters, getattr(params, field_name))
        assert wavelength_params.wavelength == wavelength_dict["wavelength"]
        assert wavelength_params.ls == wavelength_dict["ls"]
        assert wavelength_params.beaddiam == wavelength_dict.get("beaddiam")  # Optional
        assert wavelength_params.otf_path == Path(str(wavelength_dict["otf_path"]))
    assert params.sim_otf_params.model_dump() == SIMOTFParameters().model_dump()
    assert params.sim_recon_params.model_dump() == SIMReconParameters().model_dump()

    # Set up and run the service
    service = SIMReconService(environment={"queue": ""}, transport=offline_transport)
    service.log = MagicMock()  # Mock the logger to evaluate calls
    service.initializing()
    if use_recwrap:
        recwrap = MockRW(offline_transport)
        recwrap.recipe_step = {"parameters": pysimrecon_test_message}
    else:
        recwrap = None
    service.call_pysimrecon(
        recwrap,
        header=header,
        message=pysimrecon_test_message,
    )

    # Check that the main block in the function was run
    service.log.info.assert_any_call(
        "Running PySIMRecon with the following parameters:\n"
        f"{json.dumps(params.model_dump(), indent=2, default=str)}"
    )
    # Check that the config files were created
    setup_dir = visit_dir / "setup"
    for file_stem in (
        "452.cfg",
        "525.cfg",
        "605.cfg",
        "655.cfg",
        "defaults.cfg",
        "config.ini",
    ):
        assert (setup_dir / file_stem).exists() and (setup_dir / file_stem).is_file()
    # Check that the message was acked
    offline_transport.ack.assert_called_once()
