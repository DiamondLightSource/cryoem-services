from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.clem_process_raw_lifs import ProcessRawLIFsService
from cryoemservices.util.models import MockRW


@pytest.fixture
def visit_dir(tmp_path: Path):
    visit_dir = tmp_path / "test_visit"
    visit_dir.mkdir(parents=True, exist_ok=True)
    return visit_dir


@pytest.fixture
def raw_dir(visit_dir: Path):
    raw_dir = visit_dir / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


@pytest.fixture
def lif_file(raw_dir: Path):
    lif_file = raw_dir / "test_file.lif"
    lif_file.touch(exist_ok=True)
    return lif_file


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "nack")
    return transport


lif_to_stack_params_matrix = (
    # Use Recipe Wrapper?
    (True,),
    (False,),
)


@pytest.mark.parametrize("test_params", lif_to_stack_params_matrix)
@mock.patch("cryoemservices.services.clem_process_raw_lifs.process_lif_file")
def test_lif_to_stack_service(
    mock_convert,
    test_params: tuple[bool],
    lif_file: Path,
    raw_dir: Path,
    offline_transport: OfflineTransport,
):
    """
    Sends a test message to the LIF processing service, which should execute the
    function with the parameters present in the message, then send messages with
    the expected outputs back to Murfey.
    """

    # Unpack test params
    (use_recwrap,) = test_params

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    lif_to_stack_test_message = {
        "lif_file": str(lif_file),
        "root_folder": raw_dir.stem,
    }

    # Set up the expected mock values
    # mock_convert.return_value = processing_results
    dummy_results = [{"series_name": f"dummy_{i}"} for i in range(4)]
    mock_convert.return_value = dummy_results

    # Set up and run the service
    service = ProcessRawLIFsService(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    if use_recwrap:
        recwrap = MockRW(offline_transport)
        recwrap.recipe_step = {"parameters": lif_to_stack_test_message}
        service.call_process_raw_lifs(
            recwrap,
            header=header,
            message=None,
        )
    else:
        service.call_process_raw_lifs(
            None,
            header=header,
            message=lif_to_stack_test_message,
        )

    # Check that the expected calls are made
    mock_convert.assert_called_with(
        file=lif_file,
        root_folder=raw_dir.stem,
        number_of_processes=20,
    )
    for result in dummy_results:
        offline_transport.send.assert_any_call(
            "murfey_feedback",
            {
                "register": "clem.register_preprocessing_result",
                "result": result,
            },
        )


def test_lif_to_stack_bad_messsage(
    offline_transport: OfflineTransport,
):
    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    bad_message = "This is a bad message"

    # Set up and run the service
    service = ProcessRawLIFsService(
        environment={"queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.call_process_raw_lifs(
        None,
        header=header,
        message=bad_message,
    )

    # Check that message was nacked with the expected parameters
    offline_transport.nack.assert_called_once_with(header)

    # Check that the message wasn't erronerously sent
    offline_transport.send.assert_not_called()


# Introduce invalid parameters field-by-field
lif_to_stack_params_bad_validation_matrix = (
    # LIF file | Root folder | Num procs
    (False, True, True),
    (True, False, True),
    (True, True, False),
)


@pytest.mark.parametrize("test_params", lif_to_stack_params_bad_validation_matrix)
def test_lif_to_stack_service_validation_failed(
    test_params: tuple[bool, bool, bool],
    lif_file: Path,
    raw_dir: Path,
    offline_transport: OfflineTransport,
):
    """
    Sends a test message to the image alignment and merging service, which should
    excecute the function using the parameters present in the message, then send
    messages with the expected outputs back to Murfey.
    """

    # Unpack test params
    (
        valid_lif_file,
        valid_root_folder,
        valid_num_procs,
    ) = test_params

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    lif_file_value: Any = str(lif_file) if valid_lif_file else 123456789
    root_folder: Any = raw_dir.stem if valid_root_folder else 123456789
    num_procs: Any = 20 if valid_num_procs else "This is a string"
    lif_to_stack_test_message = {
        "lif_file": lif_file_value,
        "root_folder": root_folder,
        "num_procs": num_procs,
    }

    # Set up and run the service
    service = ProcessRawLIFsService(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.call_process_raw_lifs(
        None,
        header=header,
        message=lif_to_stack_test_message,
    )

    # Check that the message was nacked with the expected parameters
    offline_transport.nack.assert_called_once_with(header)

    # Check that the message wasn't erronerously sent
    offline_transport.send.assert_not_called()


@mock.patch("cryoemservices.services.clem_process_raw_lifs.process_lif_file")
def test_lif_to_stack_service_processing_failed(
    mock_convert,
    lif_file: Path,
    raw_dir: Path,
    offline_transport: OfflineTransport,
):
    """
    Sends a test message to the LIF processing service, which should execute the
    function with the parameters present in the message, then send messages with
    the expected outputs back to Murfey.
    """

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    lif_to_stack_test_message = {
        "lif_file": str(lif_file),
        "root_folder": raw_dir.stem,
    }

    # Set up the expected mock values
    mock_convert.return_value = {}

    # Set up and run the service
    service = ProcessRawLIFsService(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.call_process_raw_lifs(
        None,
        header=header,
        message=lif_to_stack_test_message,
    )

    # Check that message was nacked with expected parameters
    offline_transport.nack.assert_called_once_with(header)

    # Check that the message wasn't erronerously sent
    offline_transport.send.assert_not_called()
