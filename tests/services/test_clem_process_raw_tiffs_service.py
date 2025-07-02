from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.clem_process_raw_tiffs import TIFFToStackService

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
def processing_results(processed_dir: Path, tiff_files: list[Path]):
    series_name = f"{project}--{grid}--{position.replace(' ', '_')}"

    results_dir = processed_dir / project / grid / position.replace(" ", "_")
    results_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = results_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    metadata = metadata_dir / f"{position.replace(' ', '_')}.xlif"

    results: list[dict] = []
    for c, channel in enumerate(channels):
        tiff_sublist = [str(f) for f in tiff_files if f"--C{str(c).zfill(2)}" in f.stem]
        image = results_dir / f"{channel}.tiff"
        image.touch(exist_ok=True)
        results.append(
            {
                "image_stack": str(image),
                "metadata": str(metadata),
                "series_name": series_name,
                "channel": channel,
                "number_of_members": len(channels),
                "parent_tiffs": str(tiff_sublist),
            }
        )
    return results


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@mock.patch("cryoemservices.services.clem_process_raw_tiffs.convert_tiff_to_stack")
def test_tiff_to_stack_service(
    mock_convert,
    processing_results: list[dict],
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
    tiff_to_stack_test_message = {
        "tiff_list": None,
        "tiff_file": str(tiff_files[0]),
        "root_folder": raw_dir.stem,
        "metadata": str(metadata),
    }

    # Set up expected mock values
    mock_convert.return_value = processing_results

    # Set up and run the service
    service = TIFFToStackService(environment={"queue": ""}, transport=offline_transport)
    service.initializing()
    service.call_process_raw_tiffs(
        None,
        header=header,
        message=tiff_to_stack_test_message,
    )

    # Check that the expected calls are made
    args, kwargs = mock_convert.call_args
    assert set(kwargs["tiff_list"]) == set(tiff_files)
    assert kwargs["root_folder"] == raw_dir.stem
    assert kwargs["metadata_file"] == metadata

    for result in processing_results:
        offline_transport.send.assert_any_call(
            "murfey_feedback",
            {
                "register": "clem.register_tiff_preprocessing_result",
                "result": result,
            },
        )
