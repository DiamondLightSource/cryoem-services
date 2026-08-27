import json
import subprocess
import uuid
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
    (  # Use recwrap | Stringify | Output type
        (True, None, "dv"),
        (True, None, "tiff"),
        (True, str, "dv"),
        (True, str, "tiff"),
        (True, json.dumps, "dv"),
        (True, json.dumps, "tiff"),
        (False, None, "dv"),
        (False, None, "tiff"),
        (False, str, "dv"),
        (False, str, "tiff"),
        (False, json.dumps, "dv"),
        (False, json.dumps, "tiff"),
    ),
)
def test_sim_recon_service(
    mocker: MockerFixture,
    tmp_path: Path,
    offline_transport: OfflineTransport,
    test_params: tuple[bool, Callable | None, str],
):
    # Unpack test params
    use_recwrap, func, output_type = test_params

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
        "output_type": output_type,
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

    # Mock the UUID function and its output
    uid = uuid.uuid4()
    mocker.patch("cryoemservices.services.sim_recon.uuid.uuid4", return_value=uid)

    # Set the config directory
    config_dir = visit_dir / "setup" / f"configs-{uid}"

    # Mock the subprocess call and its outputs
    file_suffix = output_type
    if file_suffix == "tiff":
        file_suffix = f"ome.{file_suffix}"
    output_file = output_dir / f"{test_file.stem}_recon.{file_suffix}"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.touch(exist_ok=True)
    stdout_lines = [
        f"INFO:sim_recon.main:Loading configurations from {config_dir / 'config.ini'}...\n",
        "INFO:sim_recon.files.config:Running with OTFs:\n",
        "452: dummy.tiff\n",
        "525: dummy.tiff\n",
        "605: dummy.tiff\n",
        "655: dummy.tiff\n",
        f"INFO:sim_recon.main:Starting reconstruction of {params.file}\n",
        f"INFO:sim_recon.recon:Reconstructed data saved to: {output_file}",
    ]
    mock_process = MagicMock()
    mock_process.stdout = stdout_lines
    mock_process.wait.return_value = 0
    mock_popen = mocker.patch(
        "cryoemservices.services.sim_recon.subprocess.Popen",
        return_value=mock_process,
    )

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

    # Log on validation success
    service.log.info.assert_any_call(
        "Received the following parameters:\n"
        f"{json.dumps(params.model_dump(), indent=2, default=str)}"
    )

    # Check that the config files were created
    for file_name in (
        "defaults.cfg",
        "452.cfg",
        "525.cfg",
        "605.cfg",
        "655.cfg",
        "config.ini",
    ):
        config_file = config_dir / file_name
        assert config_file.exists() and config_file.is_file()

    # Check that the subprocess was called with the correct parameters
    cmd = [
        "sim-recon",
        "-d",
        f"{params.file}",
        "-c",
        f"{config_dir / 'config.ini'}",
        "-o",
        f"{params.output_dir}",
        "--type",
        f"{params.output_type}",
    ]
    service.log.info(f"Running PySIMRecon with the following commands:\n{cmd}")
    mock_popen.assert_called_once_with(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    mock_process.wait.assert_called_once()

    # Check that the correct message for Murfey was constructed
    result = {
        "output_file": str(output_file),
    }
    murfey_params = {
        "register": "sim.register_reconstruction_result",
        "result": result,
    }
    service.log.info.assert_any_call(
        "Will submit the following message back to Murfey:\n"
        f"{json.dumps(murfey_params, indent=2, default=str)}"
    )

    # Check that the message was acked
    service.log.info("PySIMRecon job completed")
    offline_transport.ack.assert_called_once()
