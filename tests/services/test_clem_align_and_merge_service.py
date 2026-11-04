from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.clem_align_and_merge import (
    AlignAndMergeParameters,
    AlignAndMergeService,
    align_and_merge_stacks,
)
from cryoemservices.util.image_processing.shared import write_stack_to_tiff
from cryoemservices.util.models import MockRW
from tests.test_utils.image_processing import create_grayscale_image_with_gaussian

# Variables to use for all the tests in this module
visit_name = "cm12345-6"
project = "project_1"
position_name = "Position 1"
series_name = f"{project}/{position_name}"
channels = ("gray", "red", "green")
num_z = 5


@pytest.fixture
def series_dir(tmp_path: Path):
    series_dir = tmp_path / visit_name / "processed" / series_name
    if not series_dir.exists():
        series_dir.mkdir(parents=True)
    return series_dir


@pytest.fixture
def metadata_file(series_dir: Path):
    metadata = series_dir / "metadata" / f"{position_name}.xml"
    if not metadata.parent.exists():
        metadata.parent.mkdir(parents=True)
    if not metadata.exists():
        metadata.touch()
    return metadata


@pytest.fixture
def image_list(series_dir: Path):
    # Set up test environment
    images: list[Path] = []

    shape = (64, 64)
    dtype = "uint8"

    for c, color in enumerate(channels):
        # Create test image with desired peaks
        arr = create_grayscale_image_with_gaussian(
            shape=shape,
            num_frames=num_z,
            dtype=dtype,
            peaks=[
                {
                    "shape": shape[-2:],
                    "amplitude": 200,
                    "centre": (24 + (2 * c), 24 + (2 * c)),
                    "sigma": (3, 4),
                    "theta": 20,
                    "offset": 0,
                },
                {
                    "shape": shape[-2:],
                    "amplitude": 150,
                    "centre": (40 + (2 * c), 32 + (2 * c)),
                    "sigma": (5, 2),
                    "theta": 40,
                    "offset": 0,
                },
                {
                    "shape": shape[-2:],
                    "amplitude": 100,
                    "centre": (20 + (2 * c), 40 + (2 * c)),
                    "sigma": (5, 2),
                    "theta": 50,
                    "offset": 0,
                },
            ],
            peak_shift_per_frame=(2, -2),
            intensity_offset_per_frame=0,
        )

        # Save image to test environment
        image = write_stack_to_tiff(
            arr,
            save_dir=series_dir,
            file_name=color,
            x_res=1.0,
            y_res=1.0,
            z_res=1.0,
            units="micron",
            axes="ZYX",
            image_labels=[f"{f}" for f in range(shape[0])],
            photometric="minisblack",
        )
        images.append(image)

    return images


@pytest.mark.parametrize(
    "test_params",
    (
        # Number of images | Stringify list | Frames to crop | Align self | Flatten | Align across
        (3, False, 5, "true", "true", "true"),
        (3, True, 5, "false", "true", "true"),
        (3, False, None, "true", "true", "false"),
        (3, False, "None", "true", "true", "false"),
        (1, False, 5, "false", "false", "true"),
        (1, True, 5, "true", "true", "true"),
        (1, False, None, "false", "true", "false"),
        (1, False, "None", "false", "true", "false"),
    ),
)
def test_align_and_merge_parameters(
    test_params: tuple[
        int,
        bool,
        int | None,
        str,
        str,
        str,
    ],
    metadata_file: Path,
    image_list: list[Path],
):
    # Unpack test parameters
    num_imgs, stringify, crop_to_n_frames, align_self, flatten, align_across = (
        test_params
    )

    images = image_list[:num_imgs] if num_imgs > 1 else image_list[0]
    if stringify:
        images = (
            str([str(file) for file in images])
            if isinstance(images, list)
            else str(images)
        )

    # Construct the message to submit for testing
    params_dict = {
        "series_name": series_name,
        "images": images,
        "metadata": str(metadata_file),
        "crop_to_n_frames": crop_to_n_frames,
        "align_self": align_self,
        "flatten": flatten,
        "align_across": align_across,
    }

    # Check that validated parameters are of the allowed types/values
    validated_params = AlignAndMergeParameters(**params_dict)
    assert isinstance(validated_params.series_name, str)
    assert isinstance(validated_params.images, list)
    for file in validated_params.images:
        assert isinstance(file, Path)
    assert isinstance(validated_params.metadata, Path)
    assert (
        isinstance(validated_params.crop_to_n_frames, int)
        or validated_params.crop_to_n_frames is None
    )
    assert validated_params.align_self in (True, False)
    assert validated_params.flatten in (True, False)
    assert validated_params.align_across in (True, False)


@pytest.mark.parametrize(
    "test_params",
    (
        # Colors | Crop frames | Align self | Flatten | Align across
        (("gray", "green", "red"), 5, True, True, True),
        (("gray", "green", "red", "blue"), 5, True, True, True),
        (("gray", "green", "red"), 10, True, True, True),
        (("gray", "green", "red"), 5, False, True, False),
        (("gray", "green", "red"), None, True, True, True),
        (("gray", "green", "red"), 5, True, False, False),
        (("green", "red"), 5, True, True, True),
        (("green", "red", "blue"), 5, True, True, True),
        (("green", "red"), 10, True, True, True),
        (("green", "red"), 5, False, True, False),
        (("green", "red"), None, True, True, True),
        (("green", "red"), 5, True, False, False),
    ),
)
def test_align_and_merge_stacks(
    mocker: MockerFixture,
    test_params: tuple[tuple[str, ...], int | None, bool, bool, bool],
    metadata_file: Path,
    image_list: list[Path],
):
    # Unpack test params
    colors, crop_to_n_frames, align_self, flatten, align_across = test_params

    # Forward along only a subset of the images
    image_list = [file for file in image_list if file.stem in colors]

    # Return a mock ElementTree from 'parse'
    mock_parse = mocker.patch("cryoemservices.services.clem_align_and_merge.parse")
    mock_parse.return_value.getroot.return_value = MagicMock()

    # Mock the return value of 'get_channel_info'
    mocker.patch(
        "cryoemservices.services.clem_align_and_merge.get_channel_info",
        return_value={
            color: {
                "bit_depth": 16,
                "min": 0.0,
                "max": 65536.0,
            }
            for color in colors
        },
    )

    # Run the function
    result = align_and_merge_stacks(
        image_list,
        metadata_file,
        crop_to_n_frames=crop_to_n_frames,
        align_self=align_self,
        flatten=flatten,
        align_across=align_across,
    )

    # Validate contents of the generated result
    assert set(result["image_stacks"]) == {str(file) for file in image_list}
    assert result["align_self"] == align_self
    assert result["flatten"] == flatten
    assert result["align_across"] == align_across
    output_file = Path(result["output_file"])
    assert output_file.exists()
    thumbnail = output_file.parent / ".thumbnails" / f"{output_file.stem}.png"
    assert result["thumbnail"] == str(thumbnail)
    assert result["thumbnail_size"] == (512, 512)


@pytest.mark.parametrize(
    "test_params",
    (
        # Colors | Crop frames | Align self | Flatten | Align across
        # Wrong 'crop_frames' value
        ("5", True, True, True),
        (5.0, True, True, True),
        # Wrong 'align_self' value
        (5, None, True, True),
        (None, 1, True, True),
        (5, "enabled", True, True),
        # Wrong 'flatten' value
        (5, True, None, False),
        (5, True, 1, True),
        (10, True, "enabled", True),
        # Wrong 'align_across' value
        (5, False, True, None),
        (None, True, True, 1),
        (5, True, False, "enabled"),
    ),
)
def test_align_and_merge_stacks_wrong_params(
    test_params: tuple[
        Any,
        Any,
        Any,
        Any,
    ],
    image_list,
    series_dir,
):
    # Unpack test params
    crop_to_n_frames, align_self, flatten, align_across = test_params

    # Set up the test environment
    metadata = series_dir / "metadata" / f"{position_name}.xml"
    if not metadata.exists():
        metadata.parent.mkdir(parents=True)
        metadata.touch()

    # Run the function
    with pytest.raises(ValueError):
        align_and_merge_stacks(
            image_list,
            metadata,
            crop_to_n_frames=crop_to_n_frames,
            align_self=align_self,
            flatten=flatten,
            align_across=align_across,
        )


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "nack")
    return transport


@pytest.mark.parametrize(
    "test_params",
    (
        # Use Recipe Wrapper | Crop to N frames | Align to self | Flatten | Align across
        (True, 20, True, True, True),
        (False, 20, True, True, False),
        (False, 20, False, True, True),
        (False, 20, False, True, False),
    ),
)
def test_align_and_merge_service(
    mocker: MockerFixture,
    test_params: tuple[bool, int, bool, bool, bool],
    image_list: list[Path],
    metadata_file: Path,
    series_dir: Path,
    offline_transport: OfflineTransport,
):
    """
    Sends a test message to the image alignment and merging service, which should
    excecute the function using the parameters present in the message, then send
    messages with the expected outputs back to Murfey.
    """

    # Unpack test params
    use_recwrap, crop_to_n_frames, align_self, flatten, align_across = test_params
    num_procs = 4

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    align_and_merge_test_message = {
        "series_name": series_name,
        "images": [str(f) for f in image_list],
        "metadata": str(metadata_file),
        "crop_to_n_frames": crop_to_n_frames,
        "align_self": align_self,
        "flatten": flatten,
        "align_across": align_across,
        "num_procs": num_procs,
    }

    # Set up expected return values
    composite_image_stem = "composite"
    if any(color in channels for color in ("gray", "grey")):
        composite_image_stem += "_BF_FL"
    else:
        composite_image_stem += "_FL"
    composite_image = series_dir / f"{composite_image_stem}.tiff"
    thumbnail_image = (
        composite_image.parent / ".thumbnails" / f"{composite_image.stem}.png"
    )

    result = {
        "image_stacks": [str(f) for f in image_list],  # Convert Path to str
        "align_self": align_self,
        "flatten": flatten,
        "align_across": align_across,
        "output_file": str(composite_image),
        "thumbnail": str(thumbnail_image),
        "thumbnail_size": (512, 512),
    }

    # Mock the align and merge function
    mock_align = mocker.patch(
        "cryoemservices.services.clem_align_and_merge.align_and_merge_stacks",
        return_value=result,
    )

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
            message=None,
        )
    else:
        service.call_align_and_merge(
            None,
            header=header,
            message=align_and_merge_test_message,
        )

    # Check that the expected calls were made
    # 'align and merge' command should be called
    mock_align.assert_called_once_with(
        images=image_list,
        metadata=metadata_file,
        crop_to_n_frames=crop_to_n_frames,
        align_self=align_self,
        flatten=flatten,
        align_across=align_across,
        num_procs=num_procs,
        logger=service.log,
    )
    # 'images' service should be called
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "tiff_to_apng",
            "input_file": result["output_file"],
            "output_file": result["thumbnail"],
            "target_size": result["thumbnail_size"],
        },
    )
    # 'murfey_feedback' should be called
    offline_transport.send.assert_any_call(
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
    offline_transport.nack.assert_called_once_with(header, requeue=False)

    # Check that the message wasn't erronerously sent
    offline_transport.send.assert_not_called()


@pytest.mark.parametrize(
    "test_params",
    (  # Introduce invalid parameters field-by-field
        # Series name | Images | Metadata | Crop to N frames | Align to self | Flatten | Align across
        (False, True, True, 20, True, True, True),
        (True, False, True, 20, True, True, True),
        (True, True, False, 20, True, True, True),
        (True, True, True, "Not a number", True, True, True),
        (True, True, True, 20, "off", True, True),
        (True, True, True, 20, True, "off", True),
        (True, True, True, 20, True, True, "off"),
    ),
)
def test_align_and_merge_service_validation_failed(
    test_params: tuple[bool, bool, bool, Any, Any, Any, Any],
    image_list: list[Path],
    metadata_file: Path,
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
        [str(f) for f in image_list] if valid_image_stack else [123, 456, 789]
    )
    align_and_merge_test_message = {
        "series_name": series_name if valid_series_name else 123456789,
        "images": images,
        "metadata": str(metadata_file) if valid_metadata else 123456789,
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
    offline_transport.nack.assert_called_once_with(
        header,
        requeue=(
            True
            if any(
                isinstance(item, str)
                for item in (
                    align_self,
                    flatten,
                    align_across,
                )
            )
            else False
        ),
    )

    # Check that the message wasn't erronerously sent
    offline_transport.send.assert_not_called()


def test_align_and_merge_service_process_failed(
    mocker: MockerFixture,
    image_list: list[Path],
    metadata_file: Path,
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
        "images": [str(f) for f in image_list],
        "metadata": str(metadata_file),
        "crop_to_n_frames": 50,
        "align_self": True,
        "flatten": True,
        "align_across": True,
    }

    # Set up expected return values if the process fails
    mocker.patch(
        "cryoemservices.services.clem_align_and_merge.align_and_merge_stacks",
        return_value={},
    )

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
    offline_transport.nack.assert_called_once_with(header, requeue=True)

    # Check that the message wasn't erronerously sent
    offline_transport.send.assert_not_called()
