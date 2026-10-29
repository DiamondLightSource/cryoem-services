import argparse
import re
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
from PIL import Image


def save_overlay(
    img: np.ndarray,
    masks: list,
    mask_results: list[dict],
    hole_indexes: list[int],
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

    hole_percent = sum([mask_results[i]["area_fraction"] for i in hole_indexes])
    label = (
        f"Hole = {hole_percent:.1f}%" if len(hole_indexes) > 0 else "No hole detected"
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
    hole_indexes: list[int],
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
    data = data.copy().astype(np.float32)
    np.clip(data, a_min=mean - 3 * sdev, a_max=mean + 3 * sdev, out=data)
    data = data - data.min()
    data = data / data.max() * 255
    data = data.astype("uint8")
    return data


def read_and_mask(
    image_path: Path,
    sam_checkpoint: Path,
    binning: int = 4,
    min_area_fraction: float = 0.001,
    points_per_side: int = 32,
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
    mask_generator = set_up_sam(sam_checkpoint, points_per_side)
    masks = mask_generator.generate(img_rgb)

    if len(masks) == 0:
        print("Mask generation failed")
        return img, masks

    sized_masks = []
    for mask_m in masks:
        # Un-bin the masks
        mask_m["segmentation"] = np.array(
            Image.fromarray(mask_m["segmentation"]).resize((img.shape[1], img.shape[0]))
        )
        mask_m["area"] = mask_m["area"] * binning**2
        if mask_m["area"] > min_area_fraction * img.shape[0] * img.shape[1]:
            sized_masks.append(mask_m)
    return img, sized_masks


@lru_cache(maxsize=1)
def set_up_sam(checkpoint: Path, points_per_side: int):
    # Do the import here so the module functions can be loaded without SAM
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    # Load SAM model
    checkpoint_type = re.findall("vit_.", checkpoint.stem)
    sam = sam_model_registry[checkpoint_type[0] if checkpoint_type else "vit_h"](
        checkpoint=checkpoint
    )
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device.upper()}")
    sam.to(device=device)
    sam_mask = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side)
    return sam_mask


def filter_masks(masks: list) -> list:
    filtered_masks: list = []
    filtered_mask_sizes: list = []
    # Descending sort by the size of the mask
    mask_sizes = [m["segmentation"].sum() for m in masks]
    sorted_size_indexes = np.argsort(mask_sizes)[::-1]
    for mask_index in sorted_size_indexes:
        # Compare each mask to the ones that have been kept
        keep = True
        mask1 = masks[mask_index]["segmentation"]
        for m2_dict, m2_size in zip(filtered_masks, filtered_mask_sizes):
            mask2 = m2_dict["segmentation"]
            overlap = np.logical_and(mask1, mask2).sum()
            enclosed_fraction = overlap / mask_sizes[mask_index]
            if enclosed_fraction > 0.95 and m2_size > mask_sizes[mask_index]:
                # Remove the current mask if covered by another mask
                keep = False
                break
        if keep:
            filtered_masks.append(masks[mask_index])
            filtered_mask_sizes.append(mask_sizes[mask_index])
    return filtered_masks


def find_hole(mask_results: list, hole_threshold: float) -> list[int]:
    hole_indexes: list[int] = []

    if len(mask_results) == 1:
        if mask_results[0]["median_intensity"] >= hole_threshold:
            hole_indexes = [0]
            print("Single mask classified as HOLE")
        else:
            print("Single mask classified as NOT HOLE")

    else:
        sorted_masks = sorted(
            mask_results, key=lambda k: k["median_intensity"], reverse=True
        )
        brightest_median = sorted_masks[0]["median_intensity"]
        print(f"Brightest mask median = {brightest_median:.1f}")

        if brightest_median >= hole_threshold:
            hole_indexes.append(sorted_masks[0]["mask_id"])
            print(f"Selected hole mask = {hole_indexes[0]}")
            brightest_area_fraction = sorted_masks[0]["area_fraction"]
            print(f"Brightest mask area = {brightest_area_fraction:.2f}%")

            # If the brightest mask is very small, include the other bright areas
            for mask in sorted_masks[1:]:
                # Stop if enough area or no more hole
                if (
                    brightest_area_fraction > 1
                    or mask["median_intensity"] < hole_threshold
                ):
                    break
                # Add next mask area
                next_index = mask["mask_id"]
                print(f"Also including mask {next_index} as hole")
                hole_indexes.append(next_index)
                brightest_area_fraction += mask["area_fraction"]
                print(f"Total mask area = {brightest_area_fraction:.2f}%")
        else:
            print(f"No hole detected (all masks below threshold {hole_threshold})")
    return hole_indexes


def get_mask_summary(image_name: str, img: np.ndarray, masks: list) -> list[dict]:
    mask_results = []
    processed_img = process_array(img)
    for i, m in enumerate(masks):
        median_intensity = np.median(processed_img[m["segmentation"]])
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
    points_per_side: int,
    binning: int,
    min_area_fraction: float,
    beam_threshold: float,
    hole_threshold: float,
    save_summary: bool,
) -> tuple[dict, list]:
    print(f"\nProcessing: {impath}")
    img, masks = read_and_mask(
        impath,
        checkpoint,
        binning=binning,
        min_area_fraction=min_area_fraction,
        points_per_side=points_per_side,
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
    hole_percent = sum([mask_results[i]["area_fraction"] for i in hole_indexes])
    hole_median = np.mean(
        [mask_results[i]["median_intensity"] for i in hole_indexes] or [0]
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
        "hole_detected": len(hole_indexes) > 0,
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
        "--points_per_side",
        default=8,
        type=int,
        help="Numbers of points per image side for segment anything",
    )
    parser.add_argument(
        "--binning",
        default=4,
        type=int,
        help="Binning to apply before masking",
    )
    parser.add_argument(
        "--min_area_fraction",
        default=0.001,
        type=float,
        help="Minimum mask area percentage to keep",
    )
    parser.add_argument(
        "--beam_threshold",
        default=10,
        type=float,
        help="Intensity threshold for beam edge detection",
    )
    parser.add_argument(
        "--hole_threshold",
        default=100,
        type=float,
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
            points_per_side=args.points_per_side,
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
