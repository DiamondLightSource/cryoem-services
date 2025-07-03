from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.clem_align_and_merge import AlignAndMergeService

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
    return transport


align_and_merge_params_matrix = (
    # Crop to N frames | Align to self | Flatten | Align across
    (20, "enabled", "mean", "enabled"),
    (20, "enabled", "min", "enabled"),
    (20, "enabled", "max", "enabled"),
    (20, "enabled", "mean", ""),
    (20, "", "min", "enabled"),
    (20, "", "max", ""),
)


@pytest.mark.parametrize("test_params", align_and_merge_params_matrix)
@mock.patch("cryoemservices.services.clem_align_and_merge.align_and_merge_stacks")
def test_align_and_merge_service(
    mock_align,
    test_params: tuple[int, str, str, str],
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
    crop_to_n_frames, align_self, flatten, align_across = test_params

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
