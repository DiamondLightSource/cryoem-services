from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.clem_process_raw_lifs import LIFToStackService
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
def processed_dir(visit_dir: Path):
    processed_dir = visit_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


@pytest.fixture
def processing_results(
    processed_dir: Path,
    lif_file: Path,
):
    results: list[dict] = []
    grids = [f"grid_{n}" for n in range(4)]
    positions = [f"position_{p}" for p in range(4)]
    channels = ("gray", "red", "green")

    # Construct results
    for grid in grids:
        for position in positions:
            # Create folder for this dataset
            data_folder = processed_dir / grid / position
            data_folder.mkdir(parents=True, exist_ok=True)

            # Create metadata folder
            metadata = data_folder / "metadata" / f"{position}.xml"
            metadata.parent.mkdir(exist_ok=True)
            metadata.touch(exist_ok=True)

            # Construct series name
            series_name = f"{grid}--{position}"

            # Create image files
            for channel in channels:
                image = data_folder / f"{channel}.tiff"
                image.touch(exist_ok=True)

                # Add to list of results
                results.append(
                    {
                        "image_stack": str(image),
                        "metadata": str(metadata),
                        "series_name": series_name,
                        "channel": channel,
                        "number_of_members": len(channels),
                        "parent_lif": str(lif_file),
                    }
                )
    return results


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


lif_to_stack_params_matrix = (
    # Use Recipe Wrapper?
    (True,),
    (False,),
)


@pytest.mark.parametrize("test_params", lif_to_stack_params_matrix)
@mock.patch("cryoemservices.services.clem_process_raw_lifs.convert_lif_to_stack")
def test_lif_to_stack_service(
    mock_convert,
    test_params: tuple[bool],
    processing_results: list[dict],
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
    mock_convert.return_value = processing_results

    # Set up and run the service
    service = LIFToStackService(environment={"queue": ""}, transport=offline_transport)
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
    for result in processing_results:
        offline_transport.send.assert_any_call(
            "murfey_feedback",
            {
                "register": "clem.register_lif_preprocessing_result",
                "result": result,
            },
        )
