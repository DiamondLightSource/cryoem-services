from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.clem_align_and_merge import AlignAndMergeService
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
def processed_dir(visit_dir: Path):
    processed_dir = visit_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


@pytest.fixture
def image_dir(processed_dir: Path):
    image_dir = processed_dir / project / grid / position.replace(" ", "_")
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


@pytest.fixture
def metadata(image_dir: Path):
    metadata_dir = image_dir / "Metadata"
    metadata_dir.mkdir(exist_ok=True)

    metadata = metadata_dir / f"{position.replace(' ', '_')}.xlif"
    metadata.touch(exist_ok=True)
    return metadata


@pytest.fixture
def image_stacks(image_dir: Path):
    images: list[Path] = []
    for channel in channels:
        image = image_dir / f"{channel}.tiff"
        image.touch(exist_ok=True)
        images.append(image)
    return images


@pytest.fixture
def series_name():
    return f"{project}--{grid}--{position.replace(' ', '_')}"


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "nack")
    return transport


align_and_merge_params_matrix = (
    # Use Recipe Wrapper | Crop to N frames | Align to self | Flatten | Align across
    (True, 20, "enabled", "mean", "enabled"),
    (True, 20, "enabled", "min", "enabled"),
    (True, 20, "enabled", "max", "enabled"),
    (False, 20, "enabled", "mean", ""),
    (False, 20, "", "min", "enabled"),
    (False, 20, "", "max", ""),
)


@pytest.mark.parametrize("test_params", align_and_merge_params_matrix)
@mock.patch("cryoemservices.services.clem_align_and_merge.align_and_merge_stacks")
def test_align_and_merge_service(
    mock_align,
    test_params: tuple[bool, int, str, str, str],
    image_stacks: list[Path],
    metadata: Path,
    series_name: str,
    image_dir: Path,
    offline_transport: OfflineTransport,
):
    """
    Sends a test message to the image alignment and merging service, which should
    excecute the function using the parameters present in the message, then send
    messages with the expected outputs back to Murfey.
    """

    # Unpack test params
    use_recwrap, crop_to_n_frames, align_self, flatten, align_across = test_params

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    align_and_merge_test_message = {
        "series_name": series_name,
        "images": [str(f) for f in image_stacks],
        "metadata": str(metadata),
        "crop_to_n_frames": crop_to_n_frames,
        "align_self": align_self,
        "flatten": flatten,
        "align_across": align_across,
    }

    # Set up expected return values
    composite_image_stem = "composite"
    if any(color in channels for color in ("gray", "grey")):
        composite_image_stem += "_BF_FL"
    else:
        composite_image_stem += "_FL"
    composite_image = image_dir / f"{composite_image_stem}.tiff"

    result = {
        "image_stacks": [str(f) for f in image_stacks],  # Convert Path to str
        "align_self": align_self,
        "flatten": flatten,
        "align_across": align_across,
        "composite_image": str(composite_image),
    }
    mock_align.return_value = result
    # Series name added to results dictionary sending to Murfey
    result["series_name"] = series_name

    # Set up and run the service
    service = AlignAndMergeService(
        environment={"queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    if use_recwrap:
        recwrap = MockRW(offline_transport)
        recwrap.recipe_step = {"parameters": align_and_merge_test_message}
        service.call_align_and_merge(
            recwrap,
            header=header,
            message={},
        )
    else:
        service.call_align_and_merge(
            None,
            header=header,
            message=align_and_merge_test_message,
        )

    # Check that the expected calls were made
    mock_align.assert_called_with(
        images=image_stacks,
        metadata=metadata,
        crop_to_n_frames=crop_to_n_frames,
        align_self=align_self,
        flatten=flatten,
        align_across=align_across,
    )
    offline_transport.send.assert_called_with(
        "murfey_feedback",
        {
            "register": "clem.register_align_and_merge_result",
            "result": result,
        },
    )


def test_align_and_merge_bad_messsage(
    offline_transport: OfflineTransport,
):
    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    bad_message = "This is a bad message"

    # Set up and run the service
    service = AlignAndMergeService(
        environment={"queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.call_align_and_merge(
        None,
        header=header,
        message=bad_message,
    )

    # Check that message was nacked with the expected parameters
    offline_transport.nack.assert_called_once_with(header)


# Introduce invalid parameters field-by-field
align_and_merge_params_bad_validation_matrix = (
    # Series name | Images | Metadata | Crop to N frames | Align to self | Flatten | Align across
    (False, True, True, 20, "enabled", "mean", "enabled"),
    (True, False, True, 20, "enabled", "mean", "enabled"),
    (True, True, False, 20, "enabled", "mean", "enabled"),
    (True, True, True, "Not a number", "enabled", "mean", "enabled"),
    (True, True, True, 20, "off", "mean", "enabled"),
    (True, True, True, 20, "enabled", "off", "enabled"),
    (True, True, True, 20, "enabled", "mean", "off"),
)


@pytest.mark.parametrize("test_params", align_and_merge_params_bad_validation_matrix)
def test_align_and_merge_service_validation_failed(
    test_params: tuple[bool, bool, bool, Any, str, str, str],
    image_stacks: list[Path],
    metadata: Path,
    series_name: str,
    offline_transport: OfflineTransport,
):
    """
    Sends a test message to the image alignment and merging service, which should
    excecute the function using the parameters present in the message, then send
    messages with the expected outputs back to Murfey.
    """

    # Unpack test params
    (
        valid_series_name,
        valid_image_stack,
        valid_metadata,
        crop_to_n_frames,
        align_self,
        flatten,
        align_across,
    ) = test_params

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    # Replace with invalid values as needed
    images: list[Any] = (
        [str(f) for f in image_stacks] if valid_image_stack else [123, 456, 789]
    )
    align_and_merge_test_message = {
        "series_name": series_name if valid_series_name else 123456789,
        "images": images,
        "metadata": str(metadata) if valid_metadata else 123456789,
        "crop_to_n_frames": crop_to_n_frames,
        "align_self": align_self,
        "flatten": flatten,
        "align_across": align_across,
    }

    # Set up and run the service
    service = AlignAndMergeService(
        environment={"queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.call_align_and_merge(
        None,
        header=header,
        message=align_and_merge_test_message,
    )

    # Check that the message was nacked
    offline_transport.nack.assert_called_once_with(header)


@mock.patch("cryoemservices.services.clem_align_and_merge.align_and_merge_stacks")
def test_align_and_merge_service_process_failed(
    mock_align,
    image_stacks: list[Path],
    metadata: Path,
    series_name: str,
    offline_transport: OfflineTransport,
):
    """
    Sends a test message to the image alignment and merging service, which should
    excecute the function using the parameters present in the message, then send
    messages with the expected outputs back to Murfey.
    """

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    align_and_merge_test_message = {
        "series_name": series_name,
        "images": [str(f) for f in image_stacks],
        "metadata": str(metadata),
        "crop_to_n_frames": 50,
        "align_self": "enabled",
        "flatten": "mean",
        "align_across": "enabled",
    }

    # Set up expected return values if the process fails
    mock_align.return_value = {}

    # Set up and run the service
    service = AlignAndMergeService(
        environment={"queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.call_align_and_merge(
        None,
        header=header,
        message=align_and_merge_test_message,
    )

    # Check that the message was nacked with the expected parameters
    offline_transport.nack.assert_called_once_with(header)
