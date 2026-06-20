"""Pre-compute SAM automatic masks for sampled annotation frames.

Only processes frames that the annotation tool actually shows (sampled
every 25 frames, skipping NaN kinematics), not all frames. This reduces
processing from thousands of frames to ~25-170 per trial.

Saves per frame:
  - .npz data files (polygons, scores, areas, bboxes) in SAM/
  - Mask visualization images in SAM/masks/
  - Overlay images (masks on original frame) in SAM/overlays/

Usage:
    python -m surgical_annotator.precompute_sam                    # all datasets
    python -m surgical_annotator.precompute_sam --dataset 7DOF2024 # one dataset
    python -m surgical_annotator.precompute_sam --resume           # skip existing
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from surgical_annotator.config import DATA_6DOF, DATA_7DOF, DATA_BAPES
from surgical_annotator.sam_segmentation import _mask_to_polygon
from surgical_annotator.frame_manager import FrameManager

logger = logging.getLogger(__name__)

# Distinct colors for mask visualization (up to 20, then cycles)
_VIS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 0, 128),
    (128, 255, 0), (0, 128, 255), (255, 128, 128), (128, 255, 128), (128, 128, 255),
    (200, 100, 50), (50, 200, 100), (100, 50, 200), (200, 200, 50), (50, 200, 200),
]


def _load_mask_generator():
    """Load SAM2 automatic mask generator.

    Returns:
        SAM2AutomaticMaskGenerator instance or None if unavailable.
    """
    try:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM2 model on {device}...")

        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2.1-hiera-small"
        )

        generator = SAM2AutomaticMaskGenerator(
            model=predictor.model,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            min_mask_region_area=100,
        )

        print("SAM2 automatic mask generator loaded.")
        return generator

    except ImportError:
        print(
            "ERROR: SAM2 not installed. Install with:\n"
            "  pip install sam2 torch torchvision"
        )
        return None
    except Exception as e:
        print(f"ERROR: Failed to load SAM2: {e}")
        return None


def _discover_sampled_frames(
    fm: FrameManager,
    dataset_name: str,
) -> list[tuple[str, Path, list[Path]]]:
    """Discover sampled frames for all trials in a dataset using FrameManager.

    Only returns the frames the annotation tool actually shows (sampled
    every 25 frames, skipping NaN kinematics), not all frames.

    Args:
        fm: FrameManager instance with trials already discovered.
        dataset_name: Name of the dataset (e.g., "7DOF2024").

    Returns:
        List of (trial_id, trial_path, frame_paths) tuples.
    """
    results = []
    all_trials = fm.discover_all_trials()
    trial_names = all_trials.get(dataset_name, [])

    for trial_name in trial_names:
        trial_id = f"{dataset_name}/{trial_name}"
        trial_info = fm.get_trial_info(trial_id)
        if not trial_info:
            continue

        sampled_indices = fm.get_sampled_frames(trial_id)
        frame_paths = []
        for idx in sampled_indices:
            path = fm.get_frame_path(trial_id, idx)
            if path and path.exists():
                frame_paths.append(path)

        if frame_paths:
            results.append((trial_id, trial_info.trial_path, frame_paths))

    return results


def _get_sam_dirs(trial_path: Path) -> tuple[Path, Path, Path]:
    """Get SAM output directories for a trial, creating them if needed.

    Args:
        trial_path: Path to the trial directory.

    Returns:
        Tuple of (sam_dir, masks_dir, overlays_dir).
    """
    sam_dir = trial_path / "SAM"
    masks_dir = sam_dir / "masks"
    overlays_dir = sam_dir / "overlays"
    sam_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    overlays_dir.mkdir(exist_ok=True)
    return sam_dir, masks_dir, overlays_dir


def _frame_index_from_path(frame_path: Path) -> int:
    """Extract frame index from a frame file path.

    Args:
        frame_path: Path to frame image file.

    Returns:
        Frame index as integer.
    """
    stem = frame_path.stem
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return int(digits) if digits else 0


def _save_mask_image(
    masks_data: list[dict],
    image_shape: tuple[int, int],
    save_path: Path,
) -> None:
    """Save a visualization of all masks as a colored image.

    Each mask gets a distinct color on a black background.

    Args:
        masks_data: List of mask dicts from SAM (each has 'segmentation').
        image_shape: (height, width) of the original image.
        save_path: Path to save the PNG.
    """
    h, w = image_shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, mask_info in enumerate(masks_data):
        color = _VIS_COLORS[idx % len(_VIS_COLORS)]
        mask = mask_info["segmentation"]
        vis[mask] = color

    Image.fromarray(vis).save(save_path)


def _save_overlay_image(
    masks_data: list[dict],
    original_image: np.ndarray,
    save_path: Path,
    mask_opacity: float = 0.45,
) -> None:
    """Save masks overlaid on the original image with transparency.

    Args:
        masks_data: List of mask dicts from SAM.
        original_image: Original RGB image as numpy array (H, W, 3).
        save_path: Path to save the PNG.
        mask_opacity: Opacity of the mask overlay (0=transparent, 1=opaque).
    """
    overlay = original_image.copy().astype(np.float32)

    for idx, mask_info in enumerate(masks_data):
        color = np.array(_VIS_COLORS[idx % len(_VIS_COLORS)], dtype=np.float32)
        mask = mask_info["segmentation"]
        overlay[mask] = overlay[mask] * (1.0 - mask_opacity) + color * mask_opacity

    # Draw contours as white outlines for clarity
    try:
        import cv2
        for mask_info in masks_data:
            mask_uint8 = (mask_info["segmentation"].astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay.astype(np.uint8), contours, -1, (255, 255, 255), 1)
            overlay = overlay.astype(np.float32)
            # Re-draw contours onto float overlay
            contour_img = np.zeros_like(original_image, dtype=np.uint8)
            cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)
            contour_mask = contour_img.sum(axis=2) > 0
            overlay[contour_mask] = 255.0
    except ImportError:
        pass  # Skip contour drawing if cv2 not available

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(save_path)


def precompute_trial(
    generator,
    trial_id: str,
    trial_path: Path,
    frame_paths: list[Path],
    resume: bool = False,
) -> int:
    """Pre-compute SAM masks for all frames in a trial.

    Saves .npz data, mask images, and overlay images.

    Args:
        generator: SAM2AutomaticMaskGenerator instance.
        trial_id: Trial identifier string.
        trial_path: Path to trial directory.
        frame_paths: List of frame image paths.
        resume: If True, skip frames that already have .npz files.

    Returns:
        Number of frames processed.
    """
    import torch

    sam_dir, masks_dir, overlays_dir = _get_sam_dirs(trial_path)
    processed = 0

    for i, frame_path in enumerate(frame_paths):
        frame_idx = _frame_index_from_path(frame_path)
        npz_path = sam_dir / f"frame_{frame_idx:05d}.npz"
        mask_img_path = masks_dir / f"frame_{frame_idx:05d}.png"
        overlay_img_path = overlays_dir / f"frame_{frame_idx:05d}.png"

        if resume and npz_path.exists():
            continue

        try:
            image = np.array(Image.open(frame_path).convert("RGB"))

            with torch.inference_mode():
                masks_data = generator.generate(image)

            if not masks_data:
                # Save empty result
                np.savez_compressed(
                    npz_path,
                    polygons=np.array([], dtype=object),
                    scores=np.array([], dtype=np.float32),
                    areas=np.array([], dtype=np.int32),
                    bboxes=np.array([], dtype=np.float32).reshape(0, 4),
                )
                # Save blank images
                blank = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                Image.fromarray(blank).save(mask_img_path)
                Image.fromarray(image).save(overlay_img_path)
                processed += 1
                continue

            # Sort by area descending (largest first)
            masks_data.sort(key=lambda m: m["area"], reverse=True)

            polygons = []
            scores = []
            areas = []
            bboxes = []

            for mask_info in masks_data:
                poly = _mask_to_polygon(mask_info["segmentation"])
                if len(poly) < 3:
                    continue
                polygons.append(poly)
                scores.append(float(mask_info.get("predicted_iou", mask_info.get("stability_score", 0.0))))
                areas.append(int(mask_info["area"]))
                bbox = mask_info["bbox"]  # [x, y, w, h] from SAM
                bboxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

            # Save .npz data
            poly_array = np.empty(len(polygons), dtype=object)
            for j, p in enumerate(polygons):
                poly_array[j] = p

            np.savez_compressed(
                npz_path,
                polygons=poly_array,
                scores=np.array(scores, dtype=np.float32),
                areas=np.array(areas, dtype=np.int32),
                bboxes=np.array(bboxes, dtype=np.float32),
            )

            # Save mask visualization (colored masks on black)
            _save_mask_image(masks_data, image.shape[:2], mask_img_path)

            # Save overlay visualization (masks on original with opacity)
            _save_overlay_image(masks_data, image, overlay_img_path)

            processed += 1

        except Exception as e:
            print(f"  ERROR frame {frame_idx}: {e}")
            continue

        if (i + 1) % 50 == 0 or i == len(frame_paths) - 1:
            print(f"  {trial_id}: {i + 1}/{len(frame_paths)} frames")

    return processed


def main():
    parser = argparse.ArgumentParser(description="Pre-compute SAM masks for sampled annotation frames")
    parser.add_argument("--dataset", type=str, default=None, help="Process only this dataset (e.g. 7DOF2024)")
    parser.add_argument("--resume", action="store_true", help="Skip frames that already have .npz files")
    args = parser.parse_args()

    # Ordered by priority: primary annotation dataset first
    datasets_ordered = [
        "7DOF2024",
        "BAPES2024",
        "6DOF2023",
    ]

    valid_datasets = set(datasets_ordered)
    if args.dataset:
        if args.dataset not in valid_datasets:
            print(f"Unknown dataset: {args.dataset}. Choose from: {datasets_ordered}")
            sys.exit(1)
        datasets_ordered = [args.dataset]

    # Initialize FrameManager to discover trials and sampled frames
    fm = FrameManager()
    fm.discover_all_trials()

    generator = _load_mask_generator()
    if generator is None:
        sys.exit(1)

    total_processed = 0
    start_time = time.time()

    for dataset_name in datasets_ordered:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        trials = _discover_sampled_frames(fm, dataset_name)
        print(f"Found {len(trials)} trials")

        for trial_id, trial_path, frame_paths in trials:
            print(f"\n  {trial_id}: {len(frame_paths)} sampled frames")
            count = precompute_trial(generator, trial_id, trial_path, frame_paths, resume=args.resume)
            total_processed += count
            if count > 0:
                print(f"  -> Processed {count} frames")
            else:
                print(f"  -> All frames already computed (skipped)")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done. Processed {total_processed} sampled frames in {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
