from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.clem_process_raw_tiffs import ProcessRawTIFFsService
from cryoemservices.util.models import MockRW

project = "project_1"
grid = "grid_1"
position = "Position 1"
channels = ("gray", "red", "green")
num_z = 200


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
def image_dir(raw_dir: Path):
    image_dir = raw_dir / project / grid
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


@pytest.fixture
def metadata(image_dir: Path):
    metadata_dir = image_dir / "Metadata"
    metadata_dir.mkdir(exist_ok=True)

    metadata = metadata_dir / f"{position}.xlif"
    metadata.touch(exist_ok=True)
    return metadata


@pytest.fixture
def tiff_files(image_dir: Path):
    images: list[Path] = []
    for c, channel in enumerate(channels):
        for z in range(num_z):
            image_file = (
                image_dir / f"{position}--Z{str(z).zfill(2)}--C{str(c).zfill(2)}.tif"
            )
            image_file.touch(exist_ok=True)
            images.append(image_file)
    return images


@pytest.fixture
def processed_dir(visit_dir: Path):
    processed_dir = visit_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "nack")
    return transport


process_raw_tiffs_params_matrix = (
    # Use Recipe Wrapper?
    (True,),
    (False,),
)


@pytest.mark.parametrize("test_params", process_raw_tiffs_params_matrix)
@mock.patch("cryoemservices.services.clem_process_raw_tiffs.process_tiff_files")
def test_process_raw_tiffs_service(
    mock_convert,
    test_params: tuple[bool],
    tiff_files: list[Path],
    metadata: Path,
    raw_dir: Path,
    offline_transport: OfflineTransport,
):
    """
    Sends a test message to the TIFF processing service, which should execute the
    function with the parameters present in the message, then send messages with
    the expected outputs back to Murfey.
    """

    # Unpack test params
    (use_recwrap,) = (test_params,)

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    process_raw_tiffs_test_message = {
        "tiff_list": None,
        "tiff_file": str(tiff_files[0]),
        "root_folder": raw_dir.stem,
        "metadata": str(metadata),
    }

    # Set up expected mock values
    result = {
        "series_name": "dummy",
        "output_files": dict.fromkeys(channels, "dummy"),
        "thumbnails": dict.fromkeys(channels, "dummy"),
        "thumbnail_size": (512, 512),
    }
    mock_convert.return_value = result

    # Set up and run the service
    service = ProcessRawTIFFsService(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    if use_recwrap:
        recwrap = MockRW(offline_transport)
        recwrap.recipe_step = {"parameters": process_raw_tiffs_test_message}
        service.call_process_raw_tiffs(
            recwrap,
            header=header,
            message=None,
        )
    else:
        service.call_process_raw_tiffs(
            None,
            header=header,
            message=process_raw_tiffs_test_message,
        )

    # Check that the expected calls are made
    args, kwargs = mock_convert.call_args
    assert sorted(kwargs["tiff_list"]) == sorted(tiff_files)
    assert kwargs["root_folder"] == raw_dir.stem
    assert kwargs["metadata_file"] == metadata

    # Check that 'images' was called for each colour
    for color in cast(dict[str, Any], result["output_files"]).keys():
        offline_transport.send.assert_any_call(
            "images",
            {
                "image_command": "tiff_to_apng",
                "input_file": cast(dict[str, str], result["output_files"])[color],
                "output_file": cast(dict[str, str], result["thumbnails"])[color],
                "target_size": result["thumbnail_size"],
                "color": color,
            },
        )
    # Check that 'murfey_feedback' was called at the end
    offline_transport.send.assert_any_call(
        "murfey_feedback",
        {
            "register": "clem.register_preprocessing_result",
            "result": result,
        },
    )
    # Check that send was called the expected number of times
    assert offline_transport.send.call_count == 1 + len(channels)


def test_process_raw_tiffs_bad_messsage(
    offline_transport: OfflineTransport,
):
    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    bad_message = "This is a bad message"

    # Set up and run the service
    service = ProcessRawTIFFsService(
        environment={"queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.call_process_raw_tiffs(
        None,
        header=header,
        message=bad_message,
    )

    # Check that message was nacked with the expected parameters
    offline_transport.nack.assert_called_once_with(header)

    # Check that the message wasn't erronerously sent
    offline_transport.send.assert_not_called()


# Introduce invalid parameters field-by-field
process_raw_tiffs_params_bad_validation_matrix = (
    # TIFF list | Tiff file | Root folder | Metadata
    (False, True, True, True),
    (True, False, True, True),
    (True, True, False, True),
    (True, True, True, False),
)


@pytest.mark.parametrize("test_params", process_raw_tiffs_params_bad_validation_matrix)
def test_process_raw_tiffs_service_validation_failed(
    test_params: tuple[bool, bool, bool, bool],
    tiff_files: list[Path],
    metadata: Path,
    raw_dir: Path,
    offline_transport: OfflineTransport,
):
    """
    Sends a test message to the TIFF processing service, which should execute the
    function with the parameters present in the message, then send messages with
    the expected outputs back to Murfey.
    """

    # Unpack test params
    valid_tiff_list, valid_tiff_file, valid_root_folder, valid_metadata = test_params

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    # Use invalid values one-by-one
    tiff_list_value = None if valid_tiff_list else [123, 456, 789]
    tiff_file_value = str(tiff_files[0]) if valid_tiff_file else 123456789
    root_folder_value = raw_dir.stem if valid_root_folder else 123456789
    metadata_value = str(metadata) if valid_metadata else 123456789
    process_raw_tiffs_test_message = {
        "tiff_list": tiff_list_value,
        "tiff_file": tiff_file_value,
        "root_folder": root_folder_value,
        "metadata": metadata_value,
    }

    # Set up and run the service
    service = ProcessRawTIFFsService(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.call_process_raw_tiffs(
        None,
        header=header,
        message=process_raw_tiffs_test_message,
    )

    # Check that the message was nacked with the expected parameters
    offline_transport.nack.assert_called_once_with(header)

    # Check that the message wasn't erronerously sent
    offline_transport.send.assert_not_called()


@mock.patch("cryoemservices.services.clem_process_raw_tiffs.process_tiff_files")
def test_process_raw_tiffs_service_processing_failed(
    mock_convert,
    tiff_files: list[Path],
    metadata: Path,
    raw_dir: Path,
    offline_transport: OfflineTransport,
):
    """
    Sends a test message to the TIFF processing service, which should execute the
    function with the parameters present in the message, then send messages with
    the expected outputs back to Murfey.
    """

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    process_raw_tiffs_test_message = {
        "tiff_list": None,
        "tiff_file": str(tiff_files[0]),
        "root_folder": raw_dir.stem,
        "metadata": str(metadata),
    }

    # Set up expected mock values
    mock_convert.return_value = None

    # Set up and run the service
    service = ProcessRawTIFFsService(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.call_process_raw_tiffs(
        None,
        header=header,
        message=process_raw_tiffs_test_message,
    )

    # Check that the message was nacked with the expected parameters
    offline_transport.nack.assert_called_once_with(header)

    # Check that the message wasn't erronerously sent
    offline_transport.send.assert_not_called()
