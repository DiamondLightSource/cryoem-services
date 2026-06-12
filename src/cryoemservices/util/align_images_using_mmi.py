import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import SimpleITK as sitk

from cryoemservices.util.image_processing import is_image_stack

logger = logging.getLogger(__name__)


def align_images_using_mmi(
    reference_array: np.ndarray,
    moving_array: np.ndarray,
    downsample_factor: int = 2,
    sampling_fraction: float = 0.5,
    shrink_factors_per_level: list[int] = [2, 1],
    smoothing_sigmas_per_level: list[float] = [1.0, 0.5],
    num_procs: int = 1,
) -> np.ndarray:
    """
    Align images to a reference using SimpleITK's implementation of the Mattes
    Mutual Information image registration method. This workflow handles 2D images
    or image stacks in both grayscale and RGB formats.

    Currently, this method works poorly for defocused images, which can lead to a
    lot of jitter in the beginning and tail frames if the images being aligned are
    a defocus series.

    Parameters
    ----------
    reference_array: np.ndarray
        The image being used as a reference.

    moving_array: np.ndarray
        The image to align.

    downsample_factor: int = 2
        The degree of binning to apply to the image during the registration process.
        While resizing, SITK preserves the dimensions of the image when it is first
        converted into an SITK Image, and will adjust the pixel size associated with
        the Image when it is subsequently resized.

    sampling_fraction: float = 0.5,
        The fraction of pixels to sample when calculating the transformation matrix.

    shrink_factors_per_level: list[int] = [2, 1]
        The degree of shrinking to apply to the image per pyramid level, with the
        registration being repeated to fine-tune the transformation matrix.

    smoothing_sigmas_per_level: list[float] = [1.0, 0.5]
        The intensity of the Gaussian blurring to apply at each pyramid level.

    num_procs: int = 1
        The number of threads to run this function with. The code has been optimised
    """

    def _register_frame(frame_num: int, ref: np.ndarray, mov: np.ndarray):
        try:
            logger.info("Setting up SITK image objects")
            ref_sitk = sitk.Cast(sitk.GetImageFromArray(ref), sitk.sitkFloat32)
            mov_sitk = sitk.Cast(sitk.GetImageFromArray(mov), sitk.sitkFloat32)
            # Downsample the frame
            if downsample_factor > 1:
                ref_small = sitk.Shrink(ref_sitk, [downsample_factor] * 2)
                mov_small = sitk.Shrink(mov_sitk, [downsample_factor] * 2)
            else:
                ref_small, mov_small = ref_sitk, mov_sitk

            # Set up registration method
            logger.info("Setting up registration method")
            registration = sitk.ImageRegistrationMethod()
            registration.SetInterpolator(sitk.sitkLinear)

            # Set the metric to use
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
            registration.SetMetricSamplingPercentage(sampling_fraction)
            registration.SetMetricSamplingStrategy(registration.RANDOM)

            # Use Mattes Mutual Information as the metric
            registration.SetOptimizerAsGradientDescentLineSearch(
                learningRate=1.0,
                numberOfIterations=200,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=5,
            )
            registration.SetOptimizerScalesFromIndexShift()

            # Register over a multi-resolution pyramid
            registration.SetShrinkFactorsPerLevel(shrink_factors_per_level)
            registration.SetSmoothingSigmasPerLevel(smoothing_sigmas_per_level)
            registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

            # Initilaise transform or reuse previous one, if available
            initial_transform = sitk.CenteredTransformInitializer(
                ref_small,
                mov_small,
                sitk.Euler2DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
            registration.SetInitialTransform(initial_transform, inPlace=False)

            # Execute registration on downsampled images
            logger.info(f"Registering frame {frame_num}")
            final_transform = registration.Execute(ref_small, mov_small)

            # Apply transform to full resolution frame
            logger.info(f"Applying transformation to frame {frame_num}")
            aligned = sitk.GetArrayFromImage(
                sitk.Resample(
                    mov_sitk,
                    transform=final_transform,
                    interpolator=sitk.sitkLinear,
                    outputPixelType=sitk.sitkFloat32,
                )
            )
            np.clip(aligned, a_min=vmin, a_max=vmax, out=aligned)
            if np.issubdtype(dtype, np.integer):
                np.rint(aligned, out=aligned)
            return aligned.astype(dtype, copy=False), frame_num
        except Exception:
            logger.warning(
                f"Error registering frame {frame_num} to reference. Returning original image",
                exc_info=True,
            )
            return mov, frame_num

    # Restrict number of threads used by SimpleITK
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    start_time = time.perf_counter()
    logger.debug(
        f"SITK image alignment settings: \n"
        f"Downsample factor: {downsample_factor}\n"
        f"Sampling percentage: {sampling_fraction}\n"
        f"Shrink factors per level: {shrink_factors_per_level}\n"
        f"Smoothing sigmas per level: {smoothing_sigmas_per_level}"
    )

    # Get initial dtype
    if reference_array.dtype != moving_array.dtype:
        logger.error("The image stacks provided do not have the same dtype")
        raise ValueError
    dtype = moving_array.dtype
    vmin, vmax = moving_array.min(), moving_array.max()

    if reference_array.shape != moving_array.shape:
        logger.error("The image stacks provided are not of the same shape")

    # Standardise frames and stacks as stacks
    if was_a_stack := is_image_stack(moving_array):
        num_frames = moving_array.shape[0]
    else:
        num_frames = 1
        reference_array = reference_array[np.newaxis, ...]
        moving_array = moving_array[np.newaxis, ...]

    # Pre-allocate output NumPy array
    aligned = np.empty(moving_array.shape, dtype=dtype)

    with ThreadPoolExecutor(max_workers=num_procs) as pool:
        futures = [
            pool.submit(
                _register_frame,
                f,
                reference_array[f],
                moving_array[f],
            )
            for f in range(num_frames)
        ]
        for future in as_completed(futures):
            frame, frame_num = future.result()
            aligned[frame_num] = frame

    # If the image was not initially a stack, flatten it
    if not was_a_stack:
        aligned = aligned[0]

    end_time = time.perf_counter()
    logger.debug(f"Completed registration of image stack in {end_time - start_time}s")
    return aligned
