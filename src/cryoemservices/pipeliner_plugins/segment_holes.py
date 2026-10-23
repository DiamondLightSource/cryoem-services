import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
import torch
from PIL import Image


def save_overlay(
    img: np.ndarray,
    masks: list,
    mask_results: list[dict],
    hole_indexes: tuple[int | None, int | None],
    beam_edge_detected: bool,
    output_figure: Path,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    img_uint8 = process_array(img)
    ax.imshow(img_uint8, cmap="gray")

    overlay = np.full((img.shape[0], img.shape[1]), np.nan, dtype=float)
    for i, m in enumerate(masks):
        overlay[m["segmentation"]] = mask_results[i]["median_intensity"]
    im = ax.imshow(
        overlay,
        cmap="turbo",
        alpha=0.5,
        vmin=min([m["median_intensity"] for m in mask_results]),
        vmax=max([m["median_intensity"] for m in mask_results]),
    )
    ax.set_title(f"Filtered Masks ({len(masks)})")
    ax.axis("off")

    hole_percent = sum(
        [mask_results[i]["area_fraction"] for i in hole_indexes if i is not None]
    )
    label = (
        f"Hole = {hole_percent:.1f}%"
        if hole_indexes[0] is not None
        else "No hole detected"
    )
    if beam_edge_detected:
        label += "\nBeam edge detected"

    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=14,
        color="white",
        verticalalignment="top",
        bbox={"facecolor": "black", "alpha": 0.7, "edgecolor": None},
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("Median Mask Intensity")
    plt.savefig(output_figure, dpi=300, bbox_inches="tight")
    plt.close()


def save_hole_mask(
    img: np.ndarray,
    masks: list,
    hole_indexes: tuple[int | None, int | None],
    output_mask: Path,
):
    hole_mask = np.zeros(img.shape, dtype=bool)
    for i in hole_indexes:
        if i is not None:
            hole_mask[masks[i]["segmentation"]] = 1
    im = Image.fromarray(hole_mask)
    im.save(output_mask)


def process_array(data: np.ndarray) -> np.ndarray:
    mean = np.mean(data)
    sdev = np.std(data)
    sigma_min = mean - 3 * sdev
    sigma_max = mean + 3 * sdev
    data = np.ndarray.copy(data)
    data[data < sigma_min] = sigma_min
    data[data > sigma_max] = sigma_max
    data = data - data.min()
    data = data / data.max() * 255
    data = data.astype("uint8")
    return data


def read_and_mask(
    image_path: Path,
    sam_checkpoint: Path,
    binning: int = 4,
    min_area_fraction: float = 0.001,
):
    if image_path.suffix == ".mrc":
        img = mrcfile.read(image_path)
    else:
        img = np.array(Image.open(image_path))

    # Apply_binning
    binned_img = process_array(
        img.reshape(img.shape[0] // binning, binning, img.shape[1] // binning, binning)
        .sum(3)
        .sum(1)
    )

    # Convert grayscale to RGB for SAM
    if img.ndim == 2:
        img_rgb = np.stack((binned_img,) * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img_rgb = np.repeat(binned_img, 3, axis=2)
    else:
        img_rgb = binned_img

    # Generate masks
    mask_generator = set_up_sam(sam_checkpoint)
    masks = mask_generator.generate(img_rgb)

    if len(masks) == 0:
        print("Mask generation failed")
        return img, masks

    masks = [
        m
        for m in masks
        if m["area"] > min_area_fraction * img.shape[0] * img.shape[1] / binning**2
    ]
    for mask_m in masks:
        # Un-bin the masks
        mask_m["segmentation"] = np.array(
            Image.fromarray(mask_m["segmentation"]).resize((img.shape[1], img.shape[0]))
        )
        mask_m["area"] = mask_m["area"] * binning**2
    return img, masks


def set_up_sam(checkpoint: Path):
    # Do the import here so the module functions can be loaded without SAM
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    # Load SAM model
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device.upper()}")
    sam.to(device=device)
    sam_mask = SamAutomaticMaskGenerator(sam)
    return sam_mask


def filter_masks(masks: list) -> list:
    filtered_masks = []
    for i, m1 in enumerate(masks):
        keep = True
        mask1 = m1["segmentation"]
        for j, m2 in enumerate(masks):
            if i == j:
                continue
            mask2 = m2["segmentation"]
            overlap = np.logical_and(mask1, mask2).sum()
            enclosed_fraction = overlap / mask1.sum()
            if enclosed_fraction > 0.95 and mask2.sum() > mask1.sum():
                keep = False
                break
        if keep:
            filtered_masks.append(m1)
    return filtered_masks


def find_hole(
    mask_results: list, hole_threshold: float
) -> tuple[int | None, int | None]:
    hole_index: int | None = None
    second_index: int | None = None

    if len(mask_results) == 1:
        if mask_results[0]["median_intensity"] >= hole_threshold:
            hole_index = 0
            print("Single mask classified as HOLE")
        else:
            print("Single mask classified as NOT HOLE")

    else:
        mask_medians = [m["median_intensity"] for m in mask_results]
        sorted_indices = np.argsort(mask_medians)[::-1]
        brightest_index: int = sorted_indices[0]
        brightest_median = mask_medians[brightest_index]
        print(f"Brightest mask median = {brightest_median:.1f}")

        if brightest_median >= hole_threshold:
            hole_index = brightest_index
            print(f"Selected hole mask = {brightest_index}")
            brightest_area_fraction = mask_results[brightest_index]["area_fraction"]
            print(f"Brightest mask area = {brightest_area_fraction:.2f}%")

            # If the brightest mask is very small, include the second brightest as well
            if len(mask_results) >= 3 and brightest_area_fraction < 0.005:
                second_index = sorted_indices[1]
                print(f"Also including mask {sorted_indices[1]} as hole")
                second_area_fraction = mask_results[sorted_indices[1]]["area_fraction"]
                print(f"Second mask area = {second_area_fraction:.2f}%")
        else:
            print(f"No hole detected (all masks below threshold {hole_threshold})")
    return hole_index, second_index


def get_mask_summary(image_name: str, img: np.ndarray, masks: list) -> list[dict]:
    mask_results = []
    for i, m in enumerate(masks):
        median_intensity = np.median(process_array(img)[m["segmentation"]])
        mask_results.append(
            {
                "image": image_name,
                "mask_id": i,
                "mask_area": m["area"],
                "median_intensity": median_intensity,
                "area_fraction": 100.0 * m["area"] / (img.shape[0] * img.shape[1]),
            }
        )
    return mask_results


def segment_single_image(
    impath: Path,
    checkpoint: Path,
    output_dir: Path,
    binning: int,
    min_area_fraction: float,
    beam_threshold: float,
    hole_threshold: float,
    save_summary: bool,
) -> tuple[dict, list]:
    print(f"\nProcessing: {impath}")
    img, masks = read_and_mask(
        impath, checkpoint, binning=binning, min_area_fraction=min_area_fraction
    )
    print(f"Generated {len(masks)} masks")
    if len(masks) == 0:
        return {}, []

    # Apply filtering
    masks = filter_masks(masks)
    print(f"Retained {len(masks)} masks after filtering")

    # Find median intensities
    mask_results = get_mask_summary(Path(impath).name, img, masks)

    # Beam edge
    beam_edge_detected = any(
        m["median_intensity"] < beam_threshold for m in mask_results
    )
    if beam_edge_detected:
        print("Beam edge detected")

    # Find hole
    hole_indexes = find_hole(mask_results, hole_threshold)
    hole_percent = sum(
        [mask_results[i]["area_fraction"] for i in hole_indexes if i is not None]
    )
    hole_median = np.mean(
        [mask_results[i]["median_intensity"] for i in hole_indexes if i is not None]
        or [0]
    )
    print(f"Hole area = {hole_percent:.2f}%")

    # Save output image
    output_mask = Path(output_dir) / f"{impath.stem}_mask.png"
    save_hole_mask(img, masks, hole_indexes, output_mask)
    if save_summary:
        output_figure = Path(output_dir) / f"{impath.stem}_overlay.png"
        save_overlay(
            img, masks, mask_results, hole_indexes, beam_edge_detected, output_figure
        )

    # Save hole percentage results
    image_results = {
        "image": impath.name,
        "hole_percent": hole_percent,
        "hole_detected": hole_indexes[0] is not None,
        "beam_edge_detected": beam_edge_detected,
        "hole_median_intensity": hole_median,
        "n_masks": len(mask_results),
    }
    return image_results, mask_results


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--micrograph_dir", required=True, help="Directory of micrographs to process"
    )
    parser.add_argument(
        "--input_glob", required=True, help="Glob string for micrographs in directory"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to store output files"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Segment anything model checkpoint"
    )
    parser.add_argument(
        "--binning", default=4, required=False, help="Binning to apply before masking"
    )
    parser.add_argument(
        "--min_area_fraction",
        default=0.001,
        required=False,
        help="Minimum mask area percentage to keep",
    )
    parser.add_argument(
        "--beam_threshold",
        default=10,
        required=False,
        help="Intensity threshold for beam edge detection",
    )
    parser.add_argument(
        "--hole_threshold",
        default=100,
        required=False,
        help="Intensity threshold for hole detection",
    )
    parser.add_argument(
        "--save_summary", action="store_true", default=False, help="Save overview image"
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)
    for impath in Path(args.micrograph_dir).glob(args.input_glob):
        image_results, mask_results = segment_single_image(
            impath=impath,
            checkpoint=Path(args.checkpoint),
            output_dir=Path(args.output_dir),
            binning=args.binning,
            min_area_fraction=args.min_area_fraction,
            beam_threshold=args.beam_threshold,
            hole_threshold=args.hole_threshold,
            save_summary=args.save_summary,
        )
        if not image_results:
            continue

        hole_df = pd.DataFrame([image_results])
        output_csv = Path(args.output_dir) / "hole_percentages.csv"
        hole_df.to_csv(
            output_csv, mode="a", header=not output_csv.exists(), index=False
        )

        # Save retained mask statistics
        mask_df = pd.DataFrame(mask_results)
        mask_csv = Path(args.output_dir) / "mask_statistics.csv"
        mask_df.to_csv(mask_csv, mode="a", header=not mask_csv.exists(), index=False)
    print(f"Results written to: {args.output_dir}")


if __name__ == "__main__":
    run()
