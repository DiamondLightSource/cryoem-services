from pathlib import Path
from unittest import mock

import mrcfile
import numpy as np
import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import easymode


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "nack")
    return transport


@mock.patch("cryoemservices.services.easymode.segment_tomogram")
@mock.patch("cryoemservices.services.easymode.load_model")
@mock.patch("cryoemservices.services.easymode.get_model")
@mock.patch("cryoemservices.services.easymode.easymode_config")
def test_easymode_service_with_mask(
    mock_easymode_config,
    mock_get_model,
    mock_load_model,
    mock_segment,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to easymode in the case with a mask
    """
    mock_get_model.side_effect = [
        ("/path/to/ribosome", {"apix": 10}),
        ("/path/to/tric", {"apix": 10}),
        ("/path/to/void", {"apix": 10}),
    ]
    mock_load_model.side_effect = ["ribosome_model", "tric_model", "void_model"]
    mock_segment.return_value = (np.random.random((10, 10, 5)), 1.5)

    input_tomogram = tmp_path / "Denoise/job007/tomograms/test_stack_aretomo.mrc"
    input_tomogram.parent.mkdir(parents=True)
    with mrcfile.new(input_tomogram) as mrc:
        mrc.set_data(np.random.random((10, 10, 5)).astype("float16"))
        mrc.header.cella = (15, 15, 7.5)
        mrc.header.mx = 10
        mrc.header.my = 10
        mrc.header.mz = 5

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    segmentation_test_message = {
        "tomogram": str(input_tomogram),
        "output_dir": f"{tmp_path}/Segmentation/job008/tomograms",
        "segmentation_apng": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo_movie.png",
        "membrain_segmentation": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo_segmented.mrc",
        "feature_list": ["ribosome", "tric"],
        "mask": "void",
        "pixel_size": "1.5",
        "batch_size": 1,
        "tta": "1",
        "relion_options": {},
    }
    Path(segmentation_test_message["output_dir"]).mkdir(parents=True)

    # Set up the mock service and send a message to it
    service = easymode.Easymode(
        environment={"queue": "", "extra_config": f"{tmp_path}/easymode_dir"},
        transport=offline_transport,
    )
    service.initializing()
    service.easymode(None, header=header, message=segmentation_test_message)

    mock_easymode_config.edit_setting.assert_called_once_with(
        "MODEL_DIRECTORY", f"{tmp_path}/easymode_dir"
    )
    mock_get_model.assert_any_call("ribosome")
    mock_get_model.assert_any_call("tric")
    mock_get_model.assert_any_call("void")
    mock_load_model.assert_any_call("/path/to/ribosome")
    mock_load_model.assert_any_call("/path/to/tric")
    mock_load_model.assert_any_call("/path/to/void")

    # Check the membrain command was run
    assert mock_segment.call_count == 3
    mock_segment.assert_any_call(
        model="ribosome_model",
        tomogram_path=str(input_tomogram),
        tta=1,
        batch_size=1,
        model_apix=10,
        input_apix=1.5,
    )
    mock_segment.assert_any_call(
        model="tric_model",
        tomogram_path=str(input_tomogram),
        tta=1,
        batch_size=1,
        model_apix=10,
        input_apix=1.5,
    )
    mock_segment.assert_any_call(
        model="void_model",
        tomogram_path=str(input_tomogram),
        tta=1,
        batch_size=1,
        model_apix=10,
        input_apix=1.5,
    )

    # Check output files were made
    assert (
        tmp_path
        / "Segmentation/job008/tomograms/test_stack_aretomo_easymode_ribosome.mrc"
    ).is_file()
    assert (
        tmp_path / "Segmentation/job008/tomograms/test_stack_aretomo_easymode_tric.mrc"
    ).is_file()
    assert (
        tmp_path / "Segmentation/job008/tomograms/test_stack_aretomo_easymode_void.mrc"
    ).is_file()

    # Check the images service request
    assert offline_transport.send.call_count == 1
    offline_transport.send.assert_any_call(
        "movie",
        {
            "image_command": "mrc_to_apng_colour",
            "file_list": [
                f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo_segmented.mrc",
                f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo_easymode_ribosome.mrc",
                f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo_easymode_tric.mrc",
            ],
            "outfile": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo_movie.png",
            "mask": str(
                f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo_easymode_void.mrc",
            ),
        },
    )


@mock.patch("cryoemservices.services.easymode.segment_tomogram")
@mock.patch("cryoemservices.services.easymode.load_model")
@mock.patch("cryoemservices.services.easymode.get_model")
@mock.patch("cryoemservices.services.easymode.tf")
def test_easymode_service_without_mask(
    mock_tensorflow,
    mock_get_model,
    mock_load_model,
    mock_segment,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to easymode for default parameters and no mask
    """
    mock_tensorflow.config.list_physical_devices.return_value = ["0"]
    mock_get_model.side_effect = [
        ("/path/to/ribosome", {"apix": 10}),
        ("/path/to/microtubule", {"apix": 20}),
        ("/path/to/tric", {"apix": 10}),
    ]
    mock_load_model.side_effect = ["ribosome_model", "microtubule_model", "tric_model"]
    mock_segment.return_value = (np.random.random((10, 10, 5)), 1.5)

    input_tomogram = tmp_path / "Denoise/job007/tomograms/test_stack_aretomo.mrc"
    input_tomogram.parent.mkdir(parents=True)
    with mrcfile.new(input_tomogram) as mrc:
        mrc.set_data(np.random.random((10, 10, 5)).astype("float16"))
        mrc.header.cella = (15, 15, 7.5)
        mrc.header.mx = 10
        mrc.header.my = 10
        mrc.header.mz = 5

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    segmentation_test_message = {
        "tomogram": str(input_tomogram),
        "output_dir": f"{tmp_path}/Segmentation/job008/tomograms",
        "segmentation_apng": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo_movie.png",
        "pixel_size": "1.5",
        "relion_options": {},
    }
    Path(segmentation_test_message["output_dir"]).mkdir(parents=True)

    # Set up the mock service and send a message to it
    service = easymode.Easymode(
        environment={"queue": "", "extra_config": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.easymode(None, header=header, message=segmentation_test_message)

    mock_tensorflow.config.list_physical_devices.assert_called_once_with("GPU")
    mock_tensorflow.config.experimental.set_memory_growth.assert_called_once_with(
        "0", True
    )

    mock_get_model.assert_any_call("ribosome")
    mock_get_model.assert_any_call("microtubule")
    mock_get_model.assert_any_call("tric")
    mock_load_model.assert_any_call("/path/to/ribosome")
    mock_load_model.assert_any_call("/path/to/microtubule")
    mock_load_model.assert_any_call("/path/to/tric")

    # Check the membrain command was run
    assert mock_segment.call_count == 3
    mock_segment.assert_any_call(
        model="ribosome_model",
        tomogram_path=str(input_tomogram),
        tta=1,
        batch_size=1,
        model_apix=10,
        input_apix=1.5,
    )
    mock_segment.assert_any_call(
        model="microtubule_model",
        tomogram_path=str(input_tomogram),
        tta=1,
        batch_size=1,
        model_apix=20,
        input_apix=1.5,
    )
    mock_segment.assert_any_call(
        model="tric_model",
        tomogram_path=str(input_tomogram),
        tta=1,
        batch_size=1,
        model_apix=10,
        input_apix=1.5,
    )

    # Check output files were made
    assert (
        tmp_path
        / "Segmentation/job008/tomograms/test_stack_aretomo_easymode_ribosome.mrc"
    ).is_file()
    assert (
        tmp_path
        / "Segmentation/job008/tomograms/test_stack_aretomo_easymode_microtubule.mrc"
    ).is_file()
    assert (
        tmp_path / "Segmentation/job008/tomograms/test_stack_aretomo_easymode_tric.mrc"
    ).is_file()

    # Check the images service request
    assert offline_transport.send.call_count == 1
    offline_transport.send.assert_any_call(
        "movie",
        {
            "image_command": "mrc_to_apng_colour",
            "file_list": [
                f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo_easymode_ribosome.mrc",
                f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo_easymode_microtubule.mrc",
                f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo_easymode_tric.mrc",
            ],
            "outfile": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo_movie.png",
            "mask": None,
        },
    )


def test_easymode_service_fail_cases(
    offline_transport,
    tmp_path,
):
    """
    Send a test message to easymode which won't run
    """

    # Set up the mock service and send an empty message to it
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    service = easymode.Easymode(
        environment={"queue": "", "extra_config": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.easymode(None, header=header, message={})

    # Check message was rejected
    assert offline_transport.nack.call_count == 1

    # Send a valid message, but with an input file which does not exist
    input_tomogram = tmp_path / "Denoise/job007/tomograms/test_stack_aretomo.mrc"
    segmentation_test_message = {
        "tomogram": str(input_tomogram),
        "output_dir": f"{tmp_path}/Segmentation/job008/tomograms",
        "segmentation_apng": f"{tmp_path}/Segmentation/job008/tomograms/test_stack_aretomo_movie.png",
        "pixel_size": "1.5",
        "relion_options": {},
    }
    service.easymode(None, header=header, message=segmentation_test_message)

    # Check message was rejected
    assert offline_transport.nack.call_count == 2
