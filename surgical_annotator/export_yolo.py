"""Export annotation JSON files to YOLO-format datasets.

Reads annotation JSONs from ``outputs/annotations/`` and produces
YOLO-format datasets suitable for training segmentation, pose/keypoint,
and detection models.

Annotation JSON → Keypoint Mapping (8 keypoints per tool)::

    Tool Anatomy:

        ════════════════  ← Shaft top line (KP 0-1)
        ║              ║
        ║    SHAFT     ║  ← Midline: avg(top[0],bot[0]) → avg(top[1],bot[1])
        ║              ║
        ════════════════  ← Shaft bottom line (KP 2-3)
               │
            Joint (KP 4) ← Shaft-to-EE transition
             /│\\
            / │ \\
       EE Left EE Right (KP 6, KP 7) ← Jaw tips
            \\ │ /
             \\│/
           EE Tip (KP 5) ← Center of jaw opening

    KP 0: Shaft top, pt 0      = lines["top"][0]
    KP 1: Shaft top, pt 1      = lines["top"][1]
    KP 2: Shaft bottom, pt 0   = lines["bottom"][0]
    KP 3: Shaft bottom, pt 1   = lines["bottom"][1]
    KP 4: Joint                = tool_joint (shaft-EE transition)
    KP 5: EE tip               = tool_ee_tip (center of jaw opening)
    KP 6: EE left              = tool_ee_left (left jaw tip)
    KP 7: EE right             = tool_ee_right (right jaw tip)

Top/bottom ordering is enforced at export time: the line with
the smaller average Y (higher in image) is always "top". The midline
(shaft centerline) is auto-computed as avg(top) → avg(bottom).

Visibility contract:
    - Annotated + visible (our vis=1)   → YOLO vis=2 → model trained, penalized
    - Annotated + occluded (our vis=0)  → YOLO vis=1 → model trained, penalized
      (model should learn to predict through occlusion, e.g. grasped peg,
       closed jaws hiding a keypoint)
    - NOT annotated (coords=null)       → YOLO vis=0 → model NOT trained
      (never guess when keypoints aren't annotated — they cannot be seen)
    - Out of scene (our vis=-1)         → YOLO vis=0 → model NOT trained

Backward Compatibility:
    For old annotations using ``tool_tooltip`` instead of ``tool_ee_tip``
    and missing ``tool_joint``:
    - ``tool_tooltip`` is used as ``ee_tip`` if ``tool_ee_tip`` is missing
    - If ``joint`` is missing, KP4 visibility is set to 0 (not labeled)

Usage::

    python -m surgical_annotator.export_yolo --task all
    python -m surgical_annotator.export_yolo --task segment
    python -m surgical_annotator.export_yolo --task pose
    python -m surgical_annotator.export_yolo --task pose_roi
    python -m surgical_annotator.export_yolo --task detect

Author: AI-ELT Project
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


# ==============================================================================
# Constants
# ==============================================================================

# Default paths
ANNOTATIONS_DIR = Path("outputs/annotations")
OUTPUT_BASE_DIR = Path("outputs/yolo_datasets")

# Default image dimensions (used only as fallback)
DEFAULT_IMG_W = 1920
DEFAULT_IMG_H = 1080

# YOLO class names
CLASS_NAMES = {0: "tool1", 1: "tool2"}

# Keypoint flip indices for horizontal flip augmentation
# Shaft top pt0 ↔ pt1, shaft bottom pt0 ↔ pt1,
# joint stays, ee_tip stays, EE left ↔ EE right
FLIP_IDX = [1, 0, 3, 2, 4, 5, 7, 6]

# Dataset base paths for resolving trial images
DATASET_BASES: dict[str, Path] = {
    "6DOF2023": Path("D:/Data/AI-ELT/6DOF2023"),
    "7DOF2024": Path("D:/Data/AI-ELT/7DOF2024"),
    "BAPES2024": Path("D:/Data/AI-ELT/BAPES2024"),
}


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class ToolAnnotation:
    """Parsed annotation for a single tool in a single frame."""

    class_id: int
    mask_polygon: list[tuple[float, float]]
    keypoints: list[tuple[float, float, int]]  # (x, y, yolo_vis) × 8
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    is_visible: bool


# ==============================================================================
# Trial / Image Resolution
# ==============================================================================


def _resolve_frame_path(trial_id: str, frame_idx: int) -> Path | None:
    """Resolve the image path for a given trial and frame index.

    Args:
        trial_id: Trial identifier (e.g. "7DOF2024/Attempt 1 - Day 1/Trial1").
        frame_idx: Frame index.

    Returns:
        Path to the image file, or None if not found.
    """
    parts = trial_id.split("/", 1)
    dataset = parts[0]
    base = DATASET_BASES.get(dataset)
    if base is None:
        return None

    trial_subpath = parts[1] if len(parts) > 1 else ""
    trial_dir = base / trial_subpath

    if dataset == "6DOF2023":
        # PNG: test{N}_{frame:04d}.png
        test_num = "".join(c for c in trial_subpath if c.isdigit())
        for pattern in [
            f"test{test_num}_{frame_idx:04d}.png",
            f"{frame_idx}.bmp",
        ]:
            p = trial_dir / pattern
            if p.exists():
                return p
    else:
        # 7DOF / BAPES: Frames/frame_{frame:05d}.bmp
        p = trial_dir / "Frames" / f"frame_{frame_idx:05d}.bmp"
        if p.exists():
            return p

    return None


def _get_image_dimensions(trial_id: str, frame_idx: int) -> tuple[int, int]:
    """Get image width and height for a frame.

    Uses PIL to read actual dimensions. Falls back to defaults.

    Args:
        trial_id: Trial identifier.
        frame_idx: Frame index.

    Returns:
        (width, height) tuple.
    """
    path = _resolve_frame_path(trial_id, frame_idx)
    if path is not None:
        try:
            with Image.open(path) as img:
                return img.size  # (width, height)
        except Exception:
            pass
    return (DEFAULT_IMG_W, DEFAULT_IMG_H)


# ==============================================================================
# Annotation Parsing
# ==============================================================================


def _our_vis_to_yolo_vis(our_vis: int) -> int:
    """Convert our visibility convention to YOLO visibility.

    Args:
        our_vis: 1=visible, 0=occluded, -1=out of scene.

    Returns:
        YOLO visibility: 2=labeled+visible, 1=labeled+occluded, 0=not labeled.
    """
    if our_vis == 1:
        return 2  # visible
    elif our_vis == 0:
        return 1  # occluded
    return 0  # out of scene → not labeled


def _extract_keypoint(
    coord: list[float] | None,
    vis_val: int,
) -> tuple[float, float, int]:
    """Extract a single keypoint with YOLO visibility.

    Visibility contract:
        - Annotated + visible (vis_val=1)  → YOLO 2: model penalized
        - Annotated + occluded (vis_val=0) → YOLO 1: model penalized
          (occluded keypoints should still be predicted by the model)
        - Not annotated (coord=None)       → YOLO 0: model NOT penalized
        - Out of scene (vis_val=-1)        → YOLO 0: model NOT penalized

    Args:
        coord: [x, y] coordinate or None if not annotated.
        vis_val: Our visibility value (1=visible, 0=occluded, -1=out of scene).

    Returns:
        (x, y, yolo_visibility). Coords are 0.0 when not annotated.
    """
    yolo_vis = _our_vis_to_yolo_vis(vis_val)
    if coord is None or not coord or vis_val == -1:
        return (0.0, 0.0, 0)
    return (float(coord[0]), float(coord[1]), yolo_vis)


def _safe_line_point(
    line: list[list[float]] | None, index: int
) -> list[float] | None:
    """Safely extract a point from a line annotation.

    Args:
        line: Line as [[x1,y1], [x2,y2]] or empty list or None.
        index: Point index (0 or 1).

    Returns:
        [x, y] point or None.
    """
    if not line or len(line) <= index:
        return None
    pt = line[index]
    if not pt or len(pt) < 2:
        return None
    return pt


def parse_frame_annotation(
    frame_data: dict[str, Any],
    tool_id: int,
) -> ToolAnnotation | None:
    """Parse annotation data for one tool in one frame.

    Args:
        frame_data: Frame annotation dict from JSON.
        tool_id: 0 for tool1, 1 for tool2.

    Returns:
        ToolAnnotation or None if tool should be skipped (no mask,
        out of scene, or skipped frame).
    """
    if frame_data.get("skipped", False):
        return None

    prefix = f"tool{tool_id + 1}"
    mask_key = f"{prefix}_mask"
    lines_key = f"{prefix}_lines"
    joint_key = f"{prefix}_joint"
    ee_tip_key = f"{prefix}_ee_tip"
    tooltip_key = f"{prefix}_tooltip"  # Legacy key for backward compatibility
    ee_left_key = f"{prefix}_ee_left"
    ee_right_key = f"{prefix}_ee_right"
    vis_key = f"{prefix}_visibility"

    # Check visibility — skip if tool is out of scene
    vis_data = frame_data.get(vis_key, {})

    # In older annotation format, visibility may use "missing" dict instead
    missing_key = f"{prefix}_missing"
    missing_data = frame_data.get(missing_key, {})

    # Determine overall tool visibility from mask visibility
    mask_vis = vis_data.get("mask", 1)  # Default visible
    if mask_vis == -1:
        return None  # Out of scene

    # Get mask polygon
    mask_polygon_raw = frame_data.get(mask_key, [])
    if not mask_polygon_raw:
        return None  # No mask → skip

    mask_polygon = [(float(pt[0]), float(pt[1])) for pt in mask_polygon_raw]

    # Compute bbox from mask polygon
    xs = [pt[0] for pt in mask_polygon]
    ys = [pt[1] for pt in mask_polygon]
    bbox = (min(xs), min(ys), max(xs), max(ys))

    # Extract lines data
    lines = frame_data.get(lines_key, {})
    top_line = lines.get("top", [])
    bottom_line = lines.get("bottom", [])

    # Enforce spatial ordering: top line = smaller avg Y (higher in image)
    if (top_line and bottom_line
            and len(top_line) >= 2 and len(bottom_line) >= 2):
        top_avg_y = (top_line[0][1] + top_line[1][1]) / 2
        bot_avg_y = (bottom_line[0][1] + bottom_line[1][1]) / 2
        if top_avg_y > bot_avg_y:
            top_line, bottom_line = bottom_line, top_line

    # Get individual keypoint visibility
    joint_vis = vis_data.get("joint", 1)
    ee_tip_vis = vis_data.get("ee_tip", vis_data.get("tooltip", 1))
    ee_left_vis = vis_data.get("ee_left", 1)
    ee_right_vis = vis_data.get("ee_right", 1)
    lines_vis = vis_data.get("lines", 1)

    # Handle backward compatibility: use tooltip as ee_tip if ee_tip missing
    ee_tip_coord = frame_data.get(ee_tip_key)
    if ee_tip_coord is None:
        ee_tip_coord = frame_data.get(tooltip_key)

    # Build 8 keypoints (top/bottom enforced by Y-sort above)
    # KP 0: Shaft top, pt 0 = lines["top"][0]
    kp0 = _extract_keypoint(_safe_line_point(top_line, 0), lines_vis)
    # KP 1: Shaft top, pt 1 = lines["top"][1]
    kp1 = _extract_keypoint(_safe_line_point(top_line, 1), lines_vis)
    # KP 2: Shaft bottom, pt 0 = lines["bottom"][0]
    kp2 = _extract_keypoint(_safe_line_point(bottom_line, 0), lines_vis)
    # KP 3: Shaft bottom, pt 1 = lines["bottom"][1]
    kp3 = _extract_keypoint(_safe_line_point(bottom_line, 1), lines_vis)
    # KP 4: Joint (shaft-to-EE transition)
    kp4 = _extract_keypoint(frame_data.get(joint_key), joint_vis)
    # KP 5: EE tip (center of jaw opening)
    kp5 = _extract_keypoint(ee_tip_coord, ee_tip_vis)
    # KP 6: EE left (left jaw tip)
    kp6 = _extract_keypoint(frame_data.get(ee_left_key), ee_left_vis)
    # KP 7: EE right (right jaw tip)
    kp7 = _extract_keypoint(frame_data.get(ee_right_key), ee_right_vis)

    keypoints = [kp0, kp1, kp2, kp3, kp4, kp5, kp6, kp7]
    is_visible = mask_vis == 1

    return ToolAnnotation(
        class_id=tool_id,
        mask_polygon=mask_polygon,
        keypoints=keypoints,
        bbox=bbox,
        is_visible=is_visible,
    )


# ==============================================================================
# Common Utilities
# ==============================================================================


def _clip_and_normalize(
    val: float, max_val: float
) -> float:
    """Clip value to [0, max_val] and normalize to [0, 1].

    Args:
        val: Raw pixel coordinate.
        max_val: Image dimension (width or height).

    Returns:
        Normalized coordinate in [0, 1].
    """
    return max(0.0, min(float(val), max_val)) / max_val


def _load_annotation_files(
    annotations_dir: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    """Load all annotation files (per-frame directories and monolithic JSONs).

    Tries per-frame directories first, then falls back to monolithic JSON
    files. Each per-frame directory is assembled into the monolithic format
    for compatibility with downstream consumers.

    Args:
        annotations_dir: Directory containing annotation files/directories.

    Returns:
        List of (path, parsed_data) tuples.
    """
    results = []
    seen_trials: set[str] = set()

    # First pass: load per-frame directories
    for trial_dir in sorted(annotations_dir.iterdir()):
        if not trial_dir.is_dir() or trial_dir.name == 'backups':
            continue

        frame_files = sorted(trial_dir.glob("frame_*.json"))
        if not frame_files:
            continue

        frames: dict[str, Any] = {}
        trial_id = ""
        for frame_file in frame_files:
            try:
                with open(frame_file, encoding="utf-8") as f:
                    data = json.load(f)
                frame_idx = data.get("frame_idx", 0)
                frames[str(frame_idx)] = data
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: skipping {frame_file}: {e}")

        if frames:
            # Get trial_id from the companion monolithic JSON if it exists
            monolithic_path = annotations_dir / f"{trial_dir.name}.json"
            trial_id = ""
            if monolithic_path.exists():
                try:
                    with open(monolithic_path, encoding="utf-8") as f:
                        mono_data = json.load(f)
                    trial_id = mono_data.get("trial_id", "")
                except (json.JSONDecodeError, OSError):
                    pass

            if not trial_id:
                # Reverse the safe_name encoding (best effort)
                trial_id = trial_dir.name

            # Repair underscore-separated trial_ids (should use '/')
            if "/" not in trial_id:
                for prefix in DATASET_PREFIXES:
                    if trial_id.startswith(prefix + "_"):
                        remainder = trial_id[len(prefix) + 1:]
                        # Find the last _Trial or _Test segment
                        for sep in ("_Trial", "_Test"):
                            idx = remainder.rfind(sep)
                            if idx >= 0:
                                mid = remainder[:idx]
                                tail = remainder[idx + 1:]
                                trial_id = f"{prefix}/{mid}/{tail}"
                                break
                        else:
                            trial_id = f"{prefix}/{remainder}"
                        break

            results.append((trial_dir, {
                "trial_id": trial_id,
                "frames": frames,
            }))
            seen_trials.add(trial_dir.name)

    # Second pass: load monolithic JSON files (skip if per-frame already loaded)
    for ann_path in sorted(annotations_dir.glob("*.json")):
        safe_name = ann_path.stem
        if safe_name in seen_trials:
            continue  # Already loaded from per-frame directory
        try:
            with open(ann_path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue  # Skip non-annotation files (e.g. cache files)
            # Repair underscore-separated trial_ids (should use '/')
            tid = data.get("trial_id", "")
            if tid and "/" not in tid:
                for prefix in DATASET_PREFIXES:
                    if tid.startswith(prefix + "_"):
                        remainder = tid[len(prefix) + 1:]
                        for sep in ("_Trial", "_Test"):
                            idx = remainder.rfind(sep)
                            if idx >= 0:
                                mid = remainder[:idx]
                                tail = remainder[idx + 1:]
                                data["trial_id"] = f"{prefix}/{mid}/{tail}"
                                break
                        else:
                            data["trial_id"] = f"{prefix}/{remainder}"
                        break
            results.append((ann_path, data))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {ann_path.name}: {e}")

    return results


def _split_frames(
    all_frames: list[dict[str, Any]],
    train_split: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split frame samples into train and val sets.

    Frames with ``exclude=True`` (e.g. RGB artifact frames) are placed
    into the training set only, keeping the validation split clean.

    Args:
        all_frames: List of frame info dicts. May contain an ``exclude`` key.
        train_split: Fraction for training (applied to non-excluded frames).
        seed: Random seed for reproducibility.

    Returns:
        (train_frames, val_frames) tuple.
    """
    normal = [f for f in all_frames if not f.get("exclude", False)]
    excluded = [f for f in all_frames if f.get("exclude", False)]

    rng = random.Random(seed)
    shuffled = list(normal)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_split)

    train = shuffled[:n_train] + excluded  # excluded go to train only
    val = shuffled[n_train:]               # val stays clean

    return train, val


def _collect_annotated_negatives(
    annotation_files: list[tuple[Path, dict[str, Any]]],
    positive_keys: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    """Find frames where both tools are out of scene (negative/background).

    Scans annotation JSONs for frames where both tools have
    ``visibility.mask == -1``, meaning neither tool is in the frame.
    These frames provide free negative training data — no annotation
    effort needed, just an empty YOLO label file.

    Args:
        annotation_files: Loaded annotation files from ``_load_annotation_files``.
        positive_keys: Set of ``(trial_id, frame_idx)`` already used as positive
            samples, to avoid duplicates.

    Returns:
        List of dicts with ``trial_id`` and ``frame_idx`` for each negative frame.
    """
    negatives: list[dict[str, Any]] = []

    for _, ann_data in annotation_files:
        trial_id = ann_data.get("trial_id", "")
        frames = ann_data.get("frames", {})

        for frame_key, frame_data in frames.items():
            frame_idx = int(frame_key)

            if frame_data.get("skipped", False):
                continue
            if frame_data.get("broken", False):
                continue  # Image may be corrupted/missing

            # Check both tools for out-of-scene visibility
            t1_vis = frame_data.get("tool1_visibility", {})
            t2_vis = frame_data.get("tool2_visibility", {})
            t1_mask = t1_vis.get("mask", 1)
            t2_mask = t2_vis.get("mask", 1)

            if t1_mask != -1 or t2_mask != -1:
                continue  # At least one tool is in scene

            if (trial_id, frame_idx) in positive_keys:
                continue  # Already a positive sample

            negatives.append({
                "trial_id": trial_id,
                "frame_idx": frame_idx,
            })

    return negatives


def _sample_negatives(
    negatives: list[dict[str, Any]],
    n_positives: int,
    max_ratio: float,
    seed: int,
    uncapped_prefixes: set[str] | None = None,
    positives_by_dataset: dict[str, int] | None = None,
    ratio_overrides: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Sample negative frames with per-dataset ratio control.

    For each dataset prefix, determines the negative budget independently:

    - Datasets in ``uncapped_prefixes`` include all available negatives.
    - Datasets with an entry in ``ratio_overrides`` use that ratio instead
      of ``max_ratio``.
    - All other datasets use ``max_ratio`` (default 1.0 = equal to positives).

    Within each capped dataset, negatives are sampled proportionally across
    trials so no single trial dominates.

    Args:
        negatives: All available negative frames.
        n_positives: Number of positive samples (global fallback).
        max_ratio: Default max negatives as fraction of positives.
        seed: Random seed for reproducibility.
        uncapped_prefixes: Dataset prefixes that bypass ratio caps.
        positives_by_dataset: Positive counts per dataset prefix. Required
            for per-dataset ratio computation; falls back to global ratio
            when absent.
        ratio_overrides: Per-dataset ratio overrides (e.g. {"BAPES2024": 3.0}).

    Returns:
        Sampled subset of negatives.
    """
    if not negatives or n_positives == 0:
        return []

    uncapped = uncapped_prefixes or set()
    overrides = ratio_overrides or {}
    pos_by_ds = positives_by_dataset or {}

    # Group negatives by dataset prefix
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for neg in negatives:
        tid = neg["trial_id"]
        ds = tid.split("/")[0] if "/" in tid else "unknown"
        by_dataset.setdefault(ds, []).append(neg)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []

    for ds, ds_negs in by_dataset.items():
        if ds in uncapped:
            # Include all negatives for this dataset
            sampled.extend(ds_negs)
            continue

        # Determine ratio and budget for this dataset
        ratio = overrides.get(ds, max_ratio)
        ds_positives = pos_by_ds.get(ds, n_positives)
        budget = int(ratio * ds_positives)
        if budget <= 0:
            continue

        if len(ds_negs) <= budget:
            sampled.extend(ds_negs)
            continue

        # Stratified sampling within this dataset (by trial)
        by_trial: dict[str, list[dict[str, Any]]] = {}
        for neg in ds_negs:
            by_trial.setdefault(neg["trial_id"], []).append(neg)

        ds_sampled: list[dict[str, Any]] = []
        for _tid, trial_negs in by_trial.items():
            proportion = len(trial_negs) / len(ds_negs)
            trial_budget = max(1, round(proportion * budget))
            rng.shuffle(trial_negs)
            ds_sampled.extend(trial_negs[:trial_budget])

        # Trim to exact budget if rounding caused overshoot
        if len(ds_sampled) > budget:
            rng.shuffle(ds_sampled)
            ds_sampled = ds_sampled[:budget]

        sampled.extend(ds_sampled)

    return sampled


def _write_negative_samples(
    negatives: list[dict[str, Any]],
    output_dir: Path,
    split: str = "train",
) -> int:
    """Copy images and write empty labels for negative (background) frames.

    Args:
        negatives: Sampled negative frames (each with trial_id, frame_idx).
        output_dir: Dataset output directory.
        split: Target split directory ("train" or "val").

    Returns:
        Number of successfully written negative samples.
    """
    written = 0
    for neg in negatives:
        trial_id = neg["trial_id"]
        frame_idx = neg["frame_idx"]
        sample_id = _make_sample_id(trial_id, frame_idx)

        src_path = _resolve_frame_path(trial_id, frame_idx)
        if src_path is None:
            continue

        ext = src_path.suffix
        dst_img = output_dir / split / "images" / f"{sample_id}{ext}"
        if not _copy_frame_image(trial_id, frame_idx, dst_img):
            continue

        # Empty label file = YOLO convention for "no objects in this image"
        label_path = output_dir / split / "labels" / f"{sample_id}.txt"
        label_path.write_text("", encoding="utf-8")
        written += 1

    return written


def _write_negative_samples_to_dir(
    negatives: list[dict[str, Any]],
    target_dir: Path,
) -> int:
    """Copy images and write empty labels for negative frames to a directory.

    Unlike ``_write_negative_samples`` which writes into a split subdirectory,
    this writes directly into ``target_dir/images/`` and ``target_dir/labels/``.

    Args:
        negatives: Sampled negative frames (each with trial_id, frame_idx).
        target_dir: Directory with ``images/`` and ``labels/`` subdirectories.

    Returns:
        Number of successfully written negative samples.
    """
    written = 0
    for neg in negatives:
        trial_id = neg["trial_id"]
        frame_idx = neg["frame_idx"]
        sample_id = _make_sample_id(trial_id, frame_idx)

        src_path = _resolve_frame_path(trial_id, frame_idx)
        if src_path is None:
            continue

        ext = src_path.suffix
        dst_img = target_dir / "images" / f"{sample_id}{ext}"
        if not _copy_frame_image(trial_id, frame_idx, dst_img):
            continue

        # Empty label file = YOLO convention for "no objects in this image"
        label_path = target_dir / "labels" / f"{sample_id}.txt"
        label_path.write_text("", encoding="utf-8")
        written += 1

    return written


def _copy_frame_image(
    trial_id: str,
    frame_idx: int,
    dst_path: Path,
) -> bool:
    """Copy a frame image to the dataset directory.

    Args:
        trial_id: Trial identifier.
        frame_idx: Frame index.
        dst_path: Destination path for the image.

    Returns:
        True if copy succeeded.
    """
    src = _resolve_frame_path(trial_id, frame_idx)
    if src is None:
        return False
    try:
        shutil.copy2(src, dst_path)
        return True
    except OSError:
        return False


def _make_sample_id(trial_id: str, frame_idx: int) -> str:
    """Create a unique filename-safe ID for a sample.

    Args:
        trial_id: Trial identifier.
        frame_idx: Frame index.

    Returns:
        Safe filename string like ``7DOF2024_Attempt1_Trial1_f00100``.
    """
    safe = trial_id.replace("/", "_").replace(" ", "_").replace("-", "_")
    return f"{safe}_f{frame_idx:05d}"


# ==============================================================================
# Cross-Validation Utilities
# ==============================================================================


DATASET_PREFIXES = ["6DOF2023", "7DOF2024", "BAPES2024"]


def _stratified_split_frames(
    all_frames: list[dict[str, Any]],
    train_split: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split frames with stratification by dataset prefix.

    Preserves per-dataset proportions in both train and val sets.
    Excluded frames go to train only. Broken frames should already
    be filtered before calling this.

    Args:
        all_frames: List of frame info dicts (must have ``trial_id``).
        train_split: Fraction for training within each group.
        seed: Random seed for reproducibility.

    Returns:
        (train_frames, val_frames) tuple.
    """
    normal = [f for f in all_frames if not f.get("exclude", False)]
    excluded = [f for f in all_frames if f.get("exclude", False)]

    # Group normal frames by dataset prefix
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for frame in normal:
        tid = frame.get("trial_id", "")
        prefix = tid.split("/")[0] if "/" in tid else "unknown"
        by_dataset.setdefault(prefix, []).append(frame)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []

    for prefix in sorted(by_dataset.keys()):
        group = list(by_dataset[prefix])
        rng.shuffle(group)
        n_train = int(len(group) * train_split)
        train.extend(group[:n_train])
        val.extend(group[n_train:])

    # Excluded frames go to train only
    train.extend(excluded)
    return train, val


def export_cv_folds(
    annotations_dir: Path = ANNOTATIONS_DIR,
    output_base: Path | None = None,
    seed: int = 42,
    include_negatives: bool = True,
    negative_ratio: float = 1.0,
) -> dict[str, Path]:
    """Export cross-validation fold datasets for segmentation.

    Creates 4 dataset directories:
        - ``fold_lodo_6DOF/``: train=7DOF+BAPES, val=6DOF
        - ``fold_lodo_7DOF/``: train=6DOF+BAPES, val=7DOF
        - ``fold_lodo_BAPES/``: train=6DOF+7DOF, val=BAPES
        - ``combined/``: train=80% all (stratified), val=20% all

    Each fold has ``data.yaml`` and ``metadata.json`` with frame counts.

    Args:
        annotations_dir: Directory with annotation JSONs.
        output_base: Base output directory. Defaults to
            ``outputs/yolo_datasets/seg_cv``.
        seed: Random seed.
        include_negatives: Whether to include negative (background) frames.
        negative_ratio: Max negatives as fraction of positives.

    Returns:
        Dict mapping fold name to data.yaml path.
    """
    if output_base is None:
        output_base = OUTPUT_BASE_DIR / "seg_cv"

    output_base.mkdir(parents=True, exist_ok=True)

    # LODO folds: leave-one-dataset-out
    lodo_folds = {
        "fold_lodo_6DOF": {
            "train": ["7DOF2024", "BAPES2024"],
            "val": ["6DOF2023"],
        },
        "fold_lodo_7DOF": {
            "train": ["6DOF2023", "BAPES2024"],
            "val": ["7DOF2024"],
        },
        "fold_lodo_BAPES": {
            "train": ["6DOF2023", "7DOF2024"],
            "val": ["BAPES2024"],
        },
    }

    results: dict[str, Path] = {}

    for fold_name, split_cfg in lodo_folds.items():
        fold_dir = output_base / fold_name
        print(f"\n{'=' * 60}")
        print(f"Exporting fold: {fold_name}")
        print(f"  Train datasets: {split_cfg['train']}")
        print(f"  Val datasets: {split_cfg['val']}")
        print(f"{'=' * 60}")

        neg_test_dir = fold_dir / "neg_test"
        data_yaml = export_segmentation_dataset(
            annotations_dir=annotations_dir,
            output_dir=fold_dir,
            seed=seed,
            include_negatives=include_negatives,
            negative_ratio=negative_ratio,
            train_datasets=split_cfg["train"],
            val_datasets=split_cfg["val"],
            neg_test_dir=neg_test_dir,
        )

        # Write metadata
        metadata = _count_fold_metadata(fold_dir, fold_name, split_cfg)
        meta_path = fold_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        results[fold_name] = data_yaml

    # Combined fold: stratified 80/20
    combined_dir = output_base / "combined"
    print(f"\n{'=' * 60}")
    print("Exporting fold: combined (stratified 80/20)")
    print(f"{'=' * 60}")

    combined_neg_test_dir = combined_dir / "neg_test"
    data_yaml = export_segmentation_dataset(
        annotations_dir=annotations_dir,
        output_dir=combined_dir,
        train_split=0.8,
        seed=seed,
        include_negatives=include_negatives,
        negative_ratio=negative_ratio,
        stratified=True,
        neg_test_dir=combined_neg_test_dir,
    )

    metadata = _count_fold_metadata(
        combined_dir, "combined",
        {"train": DATASET_PREFIXES, "val": DATASET_PREFIXES},
    )
    meta_path = combined_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results["combined"] = data_yaml

    print(f"\n{'=' * 60}")
    print(f"All CV folds exported to {output_base}")
    for name, yaml_path in results.items():
        print(f"  {name}: {yaml_path}")
    print(f"{'=' * 60}")

    return results


def _count_fold_metadata(
    fold_dir: Path,
    fold_name: str,
    split_cfg: dict[str, list[str]],
) -> dict[str, Any]:
    """Count frames in a fold directory and build metadata dict.

    Args:
        fold_dir: Fold output directory.
        fold_name: Name of the fold.
        split_cfg: Dict with ``train`` and ``val`` dataset lists.

    Returns:
        Metadata dict with frame counts.
    """
    train_imgs = list((fold_dir / "train" / "images").glob("*")) if (fold_dir / "train" / "images").exists() else []
    val_imgs = list((fold_dir / "val" / "images").glob("*")) if (fold_dir / "val" / "images").exists() else []

    train_labels = list((fold_dir / "train" / "labels").glob("*.txt")) if (fold_dir / "train" / "labels").exists() else []
    val_labels = list((fold_dir / "val" / "labels").glob("*.txt")) if (fold_dir / "val" / "labels").exists() else []

    # Count positives (non-empty labels) vs negatives (empty labels)
    train_pos = sum(1 for lf in train_labels if lf.stat().st_size > 0)
    train_neg = len(train_labels) - train_pos
    val_pos = sum(1 for lf in val_labels if lf.stat().st_size > 0)
    val_neg = len(val_labels) - val_pos

    # Count neg_test images (separate directory for FAR evaluation)
    neg_test_dir = fold_dir / "neg_test" / "images"
    neg_test_imgs = list(neg_test_dir.glob("*")) if neg_test_dir.exists() else []

    return {
        "fold_name": fold_name,
        "train_datasets": split_cfg["train"],
        "val_datasets": split_cfg["val"],
        "train_images": len(train_imgs),
        "train_positives": train_pos,
        "train_negatives": train_neg,
        "val_images": len(val_imgs),
        "val_positives": val_pos,
        "val_negatives": val_neg,
        "neg_test_images": len(neg_test_imgs),
    }


# ==============================================================================
# Segmentation Export
# ==============================================================================


def export_segmentation_dataset(
    annotations_dir: Path = ANNOTATIONS_DIR,
    output_dir: Path | None = None,
    train_split: float = 0.8,
    seed: int = 42,
    include_negatives: bool = True,
    negative_ratio: float = 1.0,
    train_datasets: list[str] | None = None,
    val_datasets: list[str] | None = None,
    stratified: bool = False,
    neg_test_dir: Path | None = None,
    uncapped_neg_datasets: set[str] | None = None,
    neg_ratio_overrides: dict[str, float] | None = None,
) -> Path:
    """Export annotations to YOLO segmentation format.

    Label format: ``class_id x1 y1 x2 y2 ... xn yn`` (normalized polygon).

    When ``include_negatives`` is True, frames where both tools are out of
    scene (``visibility.mask == -1``) are added to the training set with
    empty label files. This teaches the model background discrimination.

    **Negative routing:** Val sets receive zero negatives (positives only) so
    detection quality metrics (precision/recall/mAP) are clean. Negatives
    held out for evaluation go to ``neg_test_dir`` for separate false alarm
    rate (FAR) assessment.

    For LODO folds, negatives from ALL datasets (including held-out) go to
    train to maximize hallucination resistance; negatives matching
    ``val_datasets`` also go to ``neg_test_dir`` for FAR evaluation.

    When ``train_datasets``/``val_datasets`` are specified, samples are
    routed by trial_id prefix instead of random split. Samples whose
    trial_id starts with any entry in ``val_datasets`` go to val, those
    matching ``train_datasets`` go to train. Unmatched samples go to train.

    When ``stratified`` is True (and no dataset routing), uses
    ``_stratified_split_frames`` to preserve per-dataset proportions.

    Args:
        annotations_dir: Directory with annotation JSONs.
        output_dir: Output directory. Defaults to ``outputs/yolo_datasets/segment``.
        train_split: Train/val split ratio (used only when datasets not specified).
        seed: Random seed.
        include_negatives: Whether to include negative (background) frames.
        negative_ratio: Max negatives as fraction of positives (default 1.0).
        train_datasets: Dataset prefixes for train routing (e.g. ["7DOF2024"]).
        val_datasets: Dataset prefixes for val routing (e.g. ["6DOF2023"]).
        stratified: Use stratified split by dataset prefix (preserves proportions).
        neg_test_dir: Directory for held-out negative frames (FAR evaluation).
            When provided, negatives reserved for testing are written here
            instead of to val/. Structure: ``neg_test_dir/images/``,
            ``neg_test_dir/labels/``.
        uncapped_neg_datasets: Dataset prefixes whose negatives bypass
            the ratio cap (all negatives included).
        neg_ratio_overrides: Per-dataset negative ratio overrides
            (e.g. ``{"BAPES2024": 3.0}``).

    Returns:
        Path to generated ``data.yaml``.
    """
    if output_dir is None:
        output_dir = OUTPUT_BASE_DIR / "segment"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    if neg_test_dir is not None:
        (neg_test_dir / "images").mkdir(parents=True, exist_ok=True)
        (neg_test_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Collect all annotated frames
    all_samples: list[dict[str, Any]] = []
    annotation_files = _load_annotation_files(annotations_dir)

    for _, ann_data in annotation_files:
        trial_id = ann_data.get("trial_id", "")
        frames = ann_data.get("frames", {})

        for frame_key, frame_data in frames.items():
            frame_idx = int(frame_key)
            if frame_data.get("skipped", False):
                continue
            if frame_data.get("broken", False):
                continue  # Image may be corrupted/missing

            tools = []
            for tool_id in (0, 1):
                tool_ann = parse_frame_annotation(frame_data, tool_id)
                if tool_ann is not None:
                    tools.append(tool_ann)

            if tools:
                all_samples.append({
                    "trial_id": trial_id,
                    "frame_idx": frame_idx,
                    "tools": tools,
                    "exclude": frame_data.get("exclude", False),
                })

    if not all_samples:
        print("Warning: no annotated frames found for segmentation export")

    # Split positives: by dataset prefix, stratified, or random
    use_dataset_split = bool(train_datasets or val_datasets)
    if use_dataset_split:
        train_samples: list[dict[str, Any]] = []
        val_samples: list[dict[str, Any]] = []
        for sample in all_samples:
            tid = sample["trial_id"]
            if val_datasets and any(tid.startswith(ds) for ds in val_datasets):
                val_samples.append(sample)
            elif train_datasets and any(tid.startswith(ds) for ds in train_datasets):
                train_samples.append(sample)
            else:
                train_samples.append(sample)  # Unmatched → train
        ds_info = f" (train={train_datasets}, val={val_datasets})"
    elif stratified:
        train_samples, val_samples = _stratified_split_frames(
            all_samples, train_split, seed
        )
        ds_info = " (stratified)"
    else:
        train_samples, val_samples = _split_frames(all_samples, train_split, seed)
        ds_info = ""

    n_excluded = sum(1 for s in train_samples if s.get("exclude", False))
    n_train_normal = len(train_samples) - n_excluded

    # Collect and sample negative frames.
    # Val gets ZERO negatives (positives only for clean mAP metrics).
    # neg_test: held-out negatives for false alarm rate evaluation.
    # neg_train: all remaining negatives for training (reduces hallucinations).
    neg_train: list[dict[str, Any]] = []
    neg_test: list[dict[str, Any]] = []
    if include_negatives:
        positive_keys = {
            (s["trial_id"], s["frame_idx"]) for s in all_samples
        }
        all_negatives = _collect_annotated_negatives(
            annotation_files, positive_keys
        )
        # Compute per-dataset positive counts for ratio-based sampling
        pos_by_ds: dict[str, int] = {}
        for s in all_samples:
            ds = s["trial_id"].split("/")[0] if "/" in s["trial_id"] else "unknown"
            pos_by_ds[ds] = pos_by_ds.get(ds, 0) + 1
        neg_samples = _sample_negatives(
            all_negatives, len(all_samples), negative_ratio, seed,
            uncapped_prefixes=uncapped_neg_datasets,
            positives_by_dataset=pos_by_ds,
            ratio_overrides=neg_ratio_overrides,
        )
        if use_dataset_split:
            # LODO: negatives from val-dataset go to neg_test AND train.
            # All negatives go to train for maximum hallucination resistance.
            for neg in neg_samples:
                tid = neg["trial_id"]
                if val_datasets and any(tid.startswith(ds) for ds in val_datasets):
                    neg_test.append(neg)
                neg_train.append(neg)  # ALL negatives go to train
        else:
            # Stratified/random: split negatives into train + neg_test
            rng_neg = random.Random(seed + 1)
            neg_shuffled = list(neg_samples)
            rng_neg.shuffle(neg_shuffled)
            n_neg_train = int(len(neg_shuffled) * train_split)
            neg_train = neg_shuffled[:n_neg_train]
            neg_test = neg_shuffled[n_neg_train:]

    excl_info = f" (+{n_excluded} excluded)" if n_excluded else ""
    neg_train_info = f" (+{len(neg_train)} neg)" if neg_train else ""
    neg_test_info = f" (+{len(neg_test)} neg_test)" if neg_test else ""
    print(
        f"Segmentation{ds_info}: {n_train_normal}{excl_info}{neg_train_info} train, "
        f"{len(val_samples)} val (pos only){neg_test_info}"
    )
    if neg_train or neg_test:
        all_neg = neg_train + neg_test
        neg_by_ds: dict[str, int] = {}
        for ns in all_neg:
            ds = ns["trial_id"].split("/")[0] if "/" in ns["trial_id"] else "unknown"
            neg_by_ds[ds] = neg_by_ds.get(ds, 0) + 1
        parts = [f"{ds}={n}" for ds, n in sorted(neg_by_ds.items())]
        print(f"  Negative breakdown: {', '.join(parts)}")

    for split, samples in [("train", train_samples), ("val", val_samples)]:
        for sample in samples:
            trial_id = sample["trial_id"]
            frame_idx = sample["frame_idx"]
            sample_id = _make_sample_id(trial_id, frame_idx)
            img_w, img_h = _get_image_dimensions(trial_id, frame_idx)

            # Copy image
            src_path = _resolve_frame_path(trial_id, frame_idx)
            if src_path is None:
                continue
            ext = src_path.suffix
            dst_img = output_dir / split / "images" / f"{sample_id}{ext}"
            _copy_frame_image(trial_id, frame_idx, dst_img)

            # Write label
            label_lines = []
            for tool_ann in sample["tools"]:
                parts = [str(tool_ann.class_id)]
                for x, y in tool_ann.mask_polygon:
                    nx = _clip_and_normalize(x, img_w)
                    ny = _clip_and_normalize(y, img_h)
                    parts.extend([f"{nx:.6f}", f"{ny:.6f}"])
                label_lines.append(" ".join(parts))

            label_path = output_dir / split / "labels" / f"{sample_id}.txt"
            label_path.write_text("\n".join(label_lines), encoding="utf-8")

    # Write negative samples to train only (val stays positives-only)
    n_neg_written_train = 0
    n_neg_written_test = 0
    if neg_train:
        n_neg_written_train = _write_negative_samples(
            neg_train, output_dir, "train"
        )
    if neg_test and neg_test_dir is not None:
        n_neg_written_test = _write_negative_samples_to_dir(
            neg_test, neg_test_dir,
        )

    # Write data.yaml
    data_yaml = output_dir / "data.yaml"
    yaml_content = (
        f"path: {output_dir.absolute()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"task: segment\n"
        f"nc: 2\n"
        f"names:\n"
        f"  0: tool1\n"
        f"  1: tool2\n"
    )
    data_yaml.write_text(yaml_content, encoding="utf-8")

    print(f"Segmentation dataset written to {output_dir}")
    if n_neg_written_train or n_neg_written_test:
        print(
            f"  Negatives written: {n_neg_written_train} train, "
            f"{n_neg_written_test} neg_test"
        )
    return data_yaml


# ==============================================================================
# Pose/Keypoint Export
# ==============================================================================


def export_pose_dataset(
    annotations_dir: Path = ANNOTATIONS_DIR,
    output_dir: Path | None = None,
    use_roi_crops: bool = False,
    roi_padding: float = 0.1,
    train_split: float = 0.8,
    seed: int = 42,
) -> Path:
    """Export annotations to YOLO pose/keypoint format.

    Label format::

        class_id cx cy w h kp0_x kp0_y kp0_v ... kp6_x kp6_y kp6_v

    All coordinates are normalized to [0, 1].

    Args:
        annotations_dir: Directory with annotation JSONs.
        output_dir: Output directory. Defaults based on ``use_roi_crops``.
        use_roi_crops: If True, crop images to tool bounding box (each tool
            becomes its own sample with class_id=0). If False, full images
            with per-tool class IDs.
        roi_padding: Padding around bbox as fraction of box size (ROI mode).
        train_split: Train/val split ratio.
        seed: Random seed.

    Returns:
        Path to generated ``data.yaml``.
    """
    if output_dir is None:
        suffix = "pose_roi" if use_roi_crops else "pose"
        output_dir = OUTPUT_BASE_DIR / suffix

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Collect samples: in ROI mode each tool is a separate sample
    all_samples: list[dict[str, Any]] = []
    annotation_files = _load_annotation_files(annotations_dir)

    for _, ann_data in annotation_files:
        trial_id = ann_data.get("trial_id", "")
        frames = ann_data.get("frames", {})

        for frame_key, frame_data in frames.items():
            frame_idx = int(frame_key)
            if frame_data.get("skipped", False):
                continue

            is_excluded = frame_data.get("exclude", False)

            if use_roi_crops:
                # Each tool is a separate sample
                for tool_id in (0, 1):
                    tool_ann = parse_frame_annotation(frame_data, tool_id)
                    if tool_ann is None:
                        continue
                    # Need at least some keypoints
                    has_kp = any(kp[2] > 0 for kp in tool_ann.keypoints)
                    if not has_kp:
                        continue
                    all_samples.append({
                        "trial_id": trial_id,
                        "frame_idx": frame_idx,
                        "tool_id": tool_id,
                        "tool_ann": tool_ann,
                        "exclude": is_excluded,
                    })
            else:
                # Full image with both tools
                tools = []
                for tool_id in (0, 1):
                    tool_ann = parse_frame_annotation(frame_data, tool_id)
                    if tool_ann is not None:
                        has_kp = any(kp[2] > 0 for kp in tool_ann.keypoints)
                        if has_kp:
                            tools.append(tool_ann)
                if tools:
                    all_samples.append({
                        "trial_id": trial_id,
                        "frame_idx": frame_idx,
                        "tools": tools,
                        "exclude": is_excluded,
                    })

    train_samples, val_samples = _split_frames(all_samples, train_split, seed)
    mode_label = "Pose ROI" if use_roi_crops else "Pose"
    n_excluded = sum(1 for s in train_samples if s.get("exclude", False))
    n_train_normal = len(train_samples) - n_excluded
    excl_info = f" (+{n_excluded} excluded)" if n_excluded else ""
    print(f"{mode_label}: {n_train_normal}{excl_info} train, {len(val_samples)} val")

    for split, samples in [("train", train_samples), ("val", val_samples)]:
        for sample in samples:
            trial_id = sample["trial_id"]
            frame_idx = sample["frame_idx"]

            if use_roi_crops:
                _export_pose_roi_sample(
                    sample, split, output_dir, roi_padding
                )
            else:
                _export_pose_full_sample(sample, split, output_dir)

    # Write data.yaml
    nc = 1 if use_roi_crops else 2
    names_block = "  0: tool" if use_roi_crops else "  0: tool1\n  1: tool2"
    data_yaml = output_dir / "data.yaml"
    yaml_content = (
        f"path: {output_dir.absolute()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"task: pose\n"
        f"nc: {nc}\n"
        f"names:\n"
        f"{names_block}\n"
        f"\n"
        f"kpt_shape: [8, 3]\n"
        f"flip_idx: {FLIP_IDX}\n"
    )
    data_yaml.write_text(yaml_content, encoding="utf-8")

    print(f"{mode_label} dataset written to {output_dir}")
    return data_yaml


def _export_pose_full_sample(
    sample: dict[str, Any],
    split: str,
    output_dir: Path,
) -> None:
    """Export a full-image pose sample.

    Args:
        sample: Sample dict with trial_id, frame_idx, tools.
        split: "train" or "val".
        output_dir: Dataset output directory.
    """
    trial_id = sample["trial_id"]
    frame_idx = sample["frame_idx"]
    sample_id = _make_sample_id(trial_id, frame_idx)
    img_w, img_h = _get_image_dimensions(trial_id, frame_idx)

    src_path = _resolve_frame_path(trial_id, frame_idx)
    if src_path is None:
        return
    ext = src_path.suffix
    dst_img = output_dir / split / "images" / f"{sample_id}{ext}"
    _copy_frame_image(trial_id, frame_idx, dst_img)

    label_lines = []
    for tool_ann in sample["tools"]:
        x1, y1, x2, y2 = tool_ann.bbox
        cx = _clip_and_normalize((x1 + x2) / 2, img_w)
        cy = _clip_and_normalize((y1 + y2) / 2, img_h)
        w = _clip_and_normalize(x2 - x1, img_w)
        h = _clip_and_normalize(y2 - y1, img_h)

        parts = [
            str(tool_ann.class_id),
            f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}",
        ]

        for kp_x, kp_y, kp_v in tool_ann.keypoints:
            if kp_v > 0:
                nx = _clip_and_normalize(kp_x, img_w)
                ny = _clip_and_normalize(kp_y, img_h)
                parts.extend([f"{nx:.6f}", f"{ny:.6f}", str(kp_v)])
            else:
                parts.extend(["0.000000", "0.000000", "0"])

        label_lines.append(" ".join(parts))

    label_path = output_dir / split / "labels" / f"{sample_id}.txt"
    label_path.write_text("\n".join(label_lines), encoding="utf-8")


def _export_pose_roi_sample(
    sample: dict[str, Any],
    split: str,
    output_dir: Path,
    roi_padding: float,
) -> None:
    """Export a ROI-cropped pose sample (one tool per image).

    Args:
        sample: Sample dict with trial_id, frame_idx, tool_id, tool_ann.
        split: "train" or "val".
        output_dir: Dataset output directory.
        roi_padding: Padding fraction around bbox.
    """
    trial_id = sample["trial_id"]
    frame_idx = sample["frame_idx"]
    tool_id = sample["tool_id"]
    tool_ann: ToolAnnotation = sample["tool_ann"]

    sample_id = f"{_make_sample_id(trial_id, frame_idx)}_t{tool_id}"

    src_path = _resolve_frame_path(trial_id, frame_idx)
    if src_path is None:
        return

    try:
        img = Image.open(src_path)
    except Exception:
        return

    full_w, full_h = img.size
    x1, y1, x2, y2 = tool_ann.bbox

    # Add padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * roi_padding
    pad_y = box_h * roi_padding

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(full_w, x2 + pad_x)
    crop_y2 = min(full_h, y2 + pad_y)

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    if crop_w < 10 or crop_h < 10:
        return

    # Crop and save
    cropped = img.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))
    dst_img = output_dir / split / "images" / f"{sample_id}.jpg"
    cropped.save(str(dst_img), "JPEG", quality=95)

    # Re-normalize bbox and keypoints relative to crop
    rel_x1 = (x1 - crop_x1) / crop_w
    rel_y1 = (y1 - crop_y1) / crop_h
    rel_x2 = (x2 - crop_x1) / crop_w
    rel_y2 = (y2 - crop_y1) / crop_h

    cx = max(0.0, min(1.0, (rel_x1 + rel_x2) / 2))
    cy = max(0.0, min(1.0, (rel_y1 + rel_y2) / 2))
    w = max(0.0, min(1.0, rel_x2 - rel_x1))
    h = max(0.0, min(1.0, rel_y2 - rel_y1))

    # In ROI mode, class is always 0 (single class)
    parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]

    for kp_x, kp_y, kp_v in tool_ann.keypoints:
        if kp_v > 0:
            nx = max(0.0, min(1.0, (kp_x - crop_x1) / crop_w))
            ny = max(0.0, min(1.0, (kp_y - crop_y1) / crop_h))
            parts.extend([f"{nx:.6f}", f"{ny:.6f}", str(kp_v)])
        else:
            parts.extend(["0.000000", "0.000000", "0"])

    label_path = output_dir / split / "labels" / f"{sample_id}.txt"
    label_path.write_text(" ".join(parts), encoding="utf-8")


# ==============================================================================
# Detection Export
# ==============================================================================


def export_detection_dataset(
    annotations_dir: Path = ANNOTATIONS_DIR,
    output_dir: Path | None = None,
    train_split: float = 0.8,
    seed: int = 42,
    include_negatives: bool = True,
    negative_ratio: float = 1.0,
    uncapped_neg_datasets: set[str] | None = None,
    neg_ratio_overrides: dict[str, float] | None = None,
) -> Path:
    """Export annotations to YOLO detection format.

    Label format: ``class_id cx cy w h`` (normalized).
    Bounding box is derived from mask polygon extents.

    When ``include_negatives`` is True, frames where both tools are out of
    scene (``visibility.mask == -1``) are added to the training set with
    empty label files. This teaches the model background discrimination.

    Args:
        annotations_dir: Directory with annotation JSONs.
        output_dir: Output directory. Defaults to ``outputs/yolo_datasets/detect``.
        train_split: Train/val split ratio.
        seed: Random seed.
        include_negatives: Whether to include negative (background) frames.
        negative_ratio: Max negatives as fraction of positives (default 1.0).

    Returns:
        Path to generated ``data.yaml``.
    """
    if output_dir is None:
        output_dir = OUTPUT_BASE_DIR / "detect"

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    all_samples: list[dict[str, Any]] = []
    annotation_files = _load_annotation_files(annotations_dir)

    for _, ann_data in annotation_files:
        trial_id = ann_data.get("trial_id", "")
        frames = ann_data.get("frames", {})

        for frame_key, frame_data in frames.items():
            frame_idx = int(frame_key)
            if frame_data.get("skipped", False):
                continue

            tools = []
            for tool_id in (0, 1):
                tool_ann = parse_frame_annotation(frame_data, tool_id)
                if tool_ann is not None:
                    tools.append(tool_ann)

            if tools:
                all_samples.append({
                    "trial_id": trial_id,
                    "frame_idx": frame_idx,
                    "tools": tools,
                    "exclude": frame_data.get("exclude", False),
                })

    train_samples, val_samples = _split_frames(all_samples, train_split, seed)
    n_excluded = sum(1 for s in train_samples if s.get("exclude", False))
    n_train_normal = len(train_samples) - n_excluded

    # Collect and sample negative frames, split into train+val
    neg_train: list[dict[str, Any]] = []
    neg_val: list[dict[str, Any]] = []
    if include_negatives:
        positive_keys = {
            (s["trial_id"], s["frame_idx"]) for s in all_samples
        }
        all_negatives = _collect_annotated_negatives(
            annotation_files, positive_keys
        )
        pos_by_ds: dict[str, int] = {}
        for s in all_samples:
            ds = s["trial_id"].split("/")[0] if "/" in s["trial_id"] else "unknown"
            pos_by_ds[ds] = pos_by_ds.get(ds, 0) + 1
        neg_samples = _sample_negatives(
            all_negatives, len(all_samples), negative_ratio, seed,
            uncapped_prefixes=uncapped_neg_datasets,
            positives_by_dataset=pos_by_ds,
            ratio_overrides=neg_ratio_overrides,
        )
        # Split negatives using the same train_split ratio
        rng_neg = random.Random(seed + 1)
        neg_shuffled = list(neg_samples)
        rng_neg.shuffle(neg_shuffled)
        n_neg_train = int(len(neg_shuffled) * train_split)
        neg_train = neg_shuffled[:n_neg_train]
        neg_val = neg_shuffled[n_neg_train:]

    excl_info = f" (+{n_excluded} excluded)" if n_excluded else ""
    neg_train_info = f" (+{len(neg_train)} neg)" if neg_train else ""
    neg_val_info = f" (+{len(neg_val)} neg)" if neg_val else ""
    print(
        f"Detection: {n_train_normal}{excl_info}{neg_train_info} train, "
        f"{len(val_samples)}{neg_val_info} val"
    )
    if neg_train or neg_val:
        all_neg = neg_train + neg_val
        neg_by_ds: dict[str, int] = {}
        for ns in all_neg:
            ds = ns["trial_id"].split("/")[0] if "/" in ns["trial_id"] else "unknown"
            neg_by_ds[ds] = neg_by_ds.get(ds, 0) + 1
        parts = [f"{ds}={n}" for ds, n in sorted(neg_by_ds.items())]
        print(f"  Negative breakdown: {', '.join(parts)}")

    for split, samples in [("train", train_samples), ("val", val_samples)]:
        for sample in samples:
            trial_id = sample["trial_id"]
            frame_idx = sample["frame_idx"]
            sample_id = _make_sample_id(trial_id, frame_idx)
            img_w, img_h = _get_image_dimensions(trial_id, frame_idx)

            src_path = _resolve_frame_path(trial_id, frame_idx)
            if src_path is None:
                continue
            ext = src_path.suffix
            dst_img = output_dir / split / "images" / f"{sample_id}{ext}"
            _copy_frame_image(trial_id, frame_idx, dst_img)

            label_lines = []
            for tool_ann in sample["tools"]:
                x1, y1, x2, y2 = tool_ann.bbox
                cx = _clip_and_normalize((x1 + x2) / 2, img_w)
                cy = _clip_and_normalize((y1 + y2) / 2, img_h)
                w = _clip_and_normalize(x2 - x1, img_w)
                h = _clip_and_normalize(y2 - y1, img_h)
                label_lines.append(
                    f"{tool_ann.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )

            label_path = output_dir / split / "labels" / f"{sample_id}.txt"
            label_path.write_text("\n".join(label_lines), encoding="utf-8")

    # Write negative samples to both train and val
    n_neg_written_train = 0
    n_neg_written_val = 0
    if neg_train:
        n_neg_written_train = _write_negative_samples(
            neg_train, output_dir, "train"
        )
    if neg_val:
        n_neg_written_val = _write_negative_samples(
            neg_val, output_dir, "val"
        )

    data_yaml = output_dir / "data.yaml"
    yaml_content = (
        f"path: {output_dir.absolute()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"task: detect\n"
        f"nc: 2\n"
        f"names:\n"
        f"  0: tool1\n"
        f"  1: tool2\n"
    )
    data_yaml.write_text(yaml_content, encoding="utf-8")

    print(f"Detection dataset written to {output_dir}")
    if n_neg_written_train or n_neg_written_val:
        print(
            f"  Negatives written: {n_neg_written_train} train, "
            f"{n_neg_written_val} val"
        )
    return data_yaml


# ==============================================================================
# Peg / Pegboard Export
# ==============================================================================
#
# Peg annotation schema (per frame, see annotation_store.FrameAnnotation):
#   pegs: list of {id: 1..6, mask: [[x,y]...], keypoints: [[x,y]|null × 3],
#                  state: <one of PEG_STATES>, bbox, post_id, visible}
#   pegboard: {source_post_masks[6], source_post_keypoints[6],
#              target_post_masks[6], target_post_keypoints[6], board_mask}
#
# Object classes for the scene-level peg dataset (seg + detect):
PEG_CLASS_NAMES = {0: "peg", 1: "source_post", 2: "target_post", 3: "board"}

# Peg states where the peg is not visibly present in-frame → skip the instance.
PEG_ABSENT_STATES = {"out_of_view", "missing"}

# Peg states where the peg is present but visually hidden → keypoint vis = 1
# (model trained + penalized, mirroring the tool occlusion contract).
PEG_OCCLUDED_STATES = {"occluded"}

# Peg keypoints are 3 unlabeled landmarks (KP1/KP2/KP3). Their left/right
# semantics are NOT defined in the annotation tool, so horizontal-flip
# augmentation cannot be safely remapped — flip_idx is identity and trainers
# should disable ``fliplr`` for the peg pose model unless/until KP semantics
# are pinned down.
PEG_FLIP_IDX = [0, 1, 2]


@dataclass
class PegObject:
    """Parsed annotation for one peg/post/board instance in one frame."""

    class_id: int
    mask_polygon: list[tuple[float, float]]
    keypoints: list[tuple[float, float, int]]  # (x, y, yolo_vis); 3 for pegs
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    state: str


def _polygon_bbox(
    polygon: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Compute (x1, y1, x2, y2) extents of a polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def parse_peg_objects(
    frame_data: dict[str, Any],
    include_posts: bool = True,
    include_board: bool = True,
) -> list[PegObject]:
    """Parse peg, post, and board annotations for one frame.

    Only instances with a valid polygon mask (>= 3 vertices) are returned;
    bounding boxes are derived from the mask extents (the raw ``bbox`` field
    is ignored to avoid x,y,w,h vs x1,y1,x2,y2 ambiguity). Pegs in an
    ``out_of_view``/``missing`` state are skipped.

    Args:
        frame_data: Frame annotation dict from JSON.
        include_posts: Include source/target post instances (classes 1, 2).
        include_board: Include the board outline instance (class 3).

    Returns:
        List of ``PegObject`` (may be empty).
    """
    if frame_data.get("skipped", False) or frame_data.get("broken", False):
        return []

    objs: list[PegObject] = []

    # --- Pegs (class 0) ---
    for peg in frame_data.get("pegs", []) or []:
        if not isinstance(peg, dict):
            continue
        state = peg.get("state", "on_source_post") or "on_source_post"
        if state in PEG_ABSENT_STATES:
            continue
        mask = peg.get("mask") or []
        if len(mask) < 3:
            continue
        poly = [(float(p[0]), float(p[1])) for p in mask]
        kp_vis = 1 if state in PEG_OCCLUDED_STATES else 2
        kps: list[tuple[float, float, int]] = []
        for kp in (peg.get("keypoints") or [None, None, None]):
            if kp is None or len(kp) < 2:
                kps.append((0.0, 0.0, 0))
            else:
                kps.append((float(kp[0]), float(kp[1]), kp_vis))
        # Enforce exactly 3 keypoints.
        kps = (kps + [(0.0, 0.0, 0)] * 3)[:3]
        objs.append(PegObject(0, poly, kps, _polygon_bbox(poly), state))

    pb = frame_data.get("pegboard") or {}
    if not isinstance(pb, dict):
        pb = {}

    # --- Source / target posts (classes 1, 2) ---
    if include_posts:
        for cls_id, mask_key, kp_key in (
            (1, "source_post_masks", "source_post_keypoints"),
            (2, "target_post_masks", "target_post_keypoints"),
        ):
            masks = pb.get(mask_key) or []
            kp_list = pb.get(kp_key) or []
            for i, m in enumerate(masks):
                if not m or len(m) < 3:
                    continue
                poly = [(float(p[0]), float(p[1])) for p in m]
                kp = kp_list[i] if i < len(kp_list) else None
                if kp is None or len(kp) < 2:
                    post_kps = [(0.0, 0.0, 0)]
                else:
                    post_kps = [(float(kp[0]), float(kp[1]), 2)]
                objs.append(
                    PegObject(cls_id, poly, post_kps, _polygon_bbox(poly), "post")
                )

    # --- Board outline (class 3) ---
    if include_board:
        bm = pb.get("board_mask") or []
        if len(bm) >= 3:
            poly = [(float(p[0]), float(p[1])) for p in bm]
            objs.append(PegObject(3, poly, [], _polygon_bbox(poly), "board"))

    return objs


def _collect_peg_samples(
    annotation_files: list[tuple[Path, dict[str, Any]]],
    include_posts: bool,
    include_board: bool,
    pegs_only: bool,
) -> list[dict[str, Any]]:
    """Collect per-frame peg samples from loaded annotation files.

    Args:
        annotation_files: Output of ``_load_annotation_files``.
        include_posts: Include source/target posts (classes 1, 2).
        include_board: Include board outline (class 3).
        pegs_only: Keep only peg instances (class 0); drops posts/board even
            if collected. Used by the pose export.

    Returns:
        List of sample dicts: ``{trial_id, frame_idx, objs, exclude}``.
    """
    samples: list[dict[str, Any]] = []
    for _, ann_data in annotation_files:
        trial_id = ann_data.get("trial_id", "")
        frames = ann_data.get("frames", {})
        for frame_key, frame_data in frames.items():
            objs = parse_peg_objects(
                frame_data,
                include_posts=include_posts and not pegs_only,
                include_board=include_board and not pegs_only,
            )
            if pegs_only:
                objs = [o for o in objs if o.class_id == 0]
            if not objs:
                continue
            samples.append({
                "trial_id": trial_id,
                "frame_idx": int(frame_key),
                "objs": objs,
                "exclude": frame_data.get("exclude", False),
            })
    return samples


def export_peg_segmentation_dataset(
    annotations_dir: Path = ANNOTATIONS_DIR,
    output_dir: Path | None = None,
    train_split: float = 0.8,
    seed: int = 42,
    include_posts: bool = True,
    include_board: bool = True,
    stratified: bool = True,
) -> Path:
    """Export peg/post/board annotations to YOLO segmentation format.

    Label format: ``class_id x1 y1 ... xn yn`` (normalized polygon).
    Classes follow ``PEG_CLASS_NAMES`` (peg, source_post, target_post, board).

    Args:
        annotations_dir: Directory with annotation JSONs.
        output_dir: Output directory. Defaults to ``outputs/yolo_datasets/peg_segment``.
        train_split: Train/val split ratio.
        seed: Random seed.
        include_posts: Include source/target post masks (classes 1, 2).
        include_board: Include board outline mask (class 3).
        stratified: Split stratified by dataset prefix (recommended for the
            sparse, cross-dataset peg annotations).

    Returns:
        Path to generated ``data.yaml``.
    """
    if output_dir is None:
        output_dir = OUTPUT_BASE_DIR / "peg_segment"
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    annotation_files = _load_annotation_files(annotations_dir)
    all_samples = _collect_peg_samples(
        annotation_files, include_posts, include_board, pegs_only=False
    )
    if not all_samples:
        print("Warning: no peg-annotated frames found for segmentation export")

    if stratified:
        train_samples, val_samples = _stratified_split_frames(
            all_samples, train_split, seed
        )
    else:
        train_samples, val_samples = _split_frames(all_samples, train_split, seed)

    n_excluded = sum(1 for s in train_samples if s.get("exclude", False))
    print(
        f"Peg segmentation: {len(train_samples) - n_excluded} "
        f"(+{n_excluded} excluded) train, {len(val_samples)} val"
    )

    for split, samples in [("train", train_samples), ("val", val_samples)]:
        for sample in samples:
            trial_id = sample["trial_id"]
            frame_idx = sample["frame_idx"]
            sample_id = _make_sample_id(trial_id, frame_idx)
            img_w, img_h = _get_image_dimensions(trial_id, frame_idx)

            src_path = _resolve_frame_path(trial_id, frame_idx)
            if src_path is None:
                continue
            ext = src_path.suffix
            dst_img = output_dir / split / "images" / f"{sample_id}{ext}"
            _copy_frame_image(trial_id, frame_idx, dst_img)

            label_lines = []
            for obj in sample["objs"]:
                parts = [str(obj.class_id)]
                for x, y in obj.mask_polygon:
                    parts.append(f"{_clip_and_normalize(x, img_w):.6f}")
                    parts.append(f"{_clip_and_normalize(y, img_h):.6f}")
                label_lines.append(" ".join(parts))
            label_path = output_dir / split / "labels" / f"{sample_id}.txt"
            label_path.write_text("\n".join(label_lines), encoding="utf-8")

    names_block = "\n".join(
        f"  {cid}: {name}" for cid, name in sorted(PEG_CLASS_NAMES.items())
    )
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {output_dir.absolute()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"task: segment\n"
        f"nc: {len(PEG_CLASS_NAMES)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    print(f"Peg segmentation dataset written to {output_dir}")
    return data_yaml


def export_peg_detection_dataset(
    annotations_dir: Path = ANNOTATIONS_DIR,
    output_dir: Path | None = None,
    train_split: float = 0.8,
    seed: int = 42,
    include_posts: bool = True,
    include_board: bool = True,
    stratified: bool = True,
) -> Path:
    """Export peg/post/board annotations to YOLO detection format.

    Label format: ``class_id cx cy w h`` (normalized), bbox from mask extents.

    Args:
        annotations_dir: Directory with annotation JSONs.
        output_dir: Output directory. Defaults to ``outputs/yolo_datasets/peg_detect``.
        train_split: Train/val split ratio.
        seed: Random seed.
        include_posts: Include source/target posts (classes 1, 2).
        include_board: Include board outline (class 3).
        stratified: Split stratified by dataset prefix.

    Returns:
        Path to generated ``data.yaml``.
    """
    if output_dir is None:
        output_dir = OUTPUT_BASE_DIR / "peg_detect"
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    annotation_files = _load_annotation_files(annotations_dir)
    all_samples = _collect_peg_samples(
        annotation_files, include_posts, include_board, pegs_only=False
    )
    if not all_samples:
        print("Warning: no peg-annotated frames found for detection export")

    if stratified:
        train_samples, val_samples = _stratified_split_frames(
            all_samples, train_split, seed
        )
    else:
        train_samples, val_samples = _split_frames(all_samples, train_split, seed)

    n_excluded = sum(1 for s in train_samples if s.get("exclude", False))
    print(
        f"Peg detection: {len(train_samples) - n_excluded} "
        f"(+{n_excluded} excluded) train, {len(val_samples)} val"
    )

    for split, samples in [("train", train_samples), ("val", val_samples)]:
        for sample in samples:
            trial_id = sample["trial_id"]
            frame_idx = sample["frame_idx"]
            sample_id = _make_sample_id(trial_id, frame_idx)
            img_w, img_h = _get_image_dimensions(trial_id, frame_idx)

            src_path = _resolve_frame_path(trial_id, frame_idx)
            if src_path is None:
                continue
            ext = src_path.suffix
            dst_img = output_dir / split / "images" / f"{sample_id}{ext}"
            _copy_frame_image(trial_id, frame_idx, dst_img)

            label_lines = []
            for obj in sample["objs"]:
                x1, y1, x2, y2 = obj.bbox
                cx = _clip_and_normalize((x1 + x2) / 2, img_w)
                cy = _clip_and_normalize((y1 + y2) / 2, img_h)
                w = _clip_and_normalize(x2 - x1, img_w)
                h = _clip_and_normalize(y2 - y1, img_h)
                label_lines.append(
                    f"{obj.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )
            label_path = output_dir / split / "labels" / f"{sample_id}.txt"
            label_path.write_text("\n".join(label_lines), encoding="utf-8")

    names_block = "\n".join(
        f"  {cid}: {name}" for cid, name in sorted(PEG_CLASS_NAMES.items())
    )
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {output_dir.absolute()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"task: detect\n"
        f"nc: {len(PEG_CLASS_NAMES)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    print(f"Peg detection dataset written to {output_dir}")
    return data_yaml


def export_peg_pose_dataset(
    annotations_dir: Path = ANNOTATIONS_DIR,
    output_dir: Path | None = None,
    train_split: float = 0.8,
    seed: int = 42,
) -> Path:
    """Export peg keypoints to YOLO pose format (pegs only, single class).

    Each peg has 3 keypoints; posts and board are excluded (YOLO pose requires
    a single fixed ``kpt_shape`` across all classes). Label format::

        0 cx cy w h kp0_x kp0_y kp0_v kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v

    Args:
        annotations_dir: Directory with annotation JSONs.
        output_dir: Output directory. Defaults to ``outputs/yolo_datasets/peg_pose``.
        train_split: Train/val split ratio.
        seed: Random seed.

    Returns:
        Path to generated ``data.yaml``.
    """
    if output_dir is None:
        output_dir = OUTPUT_BASE_DIR / "peg_pose"
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    annotation_files = _load_annotation_files(annotations_dir)
    all_samples = _collect_peg_samples(
        annotation_files, include_posts=False, include_board=False, pegs_only=True
    )
    # Keep only frames where at least one peg has an annotated keypoint.
    all_samples = [
        s for s in all_samples
        if any(any(kp[2] > 0 for kp in o.keypoints) for o in s["objs"])
    ]
    if not all_samples:
        print("Warning: no peg keypoints found for pose export")

    train_samples, val_samples = _stratified_split_frames(
        all_samples, train_split, seed
    )
    n_excluded = sum(1 for s in train_samples if s.get("exclude", False))
    print(
        f"Peg pose: {len(train_samples) - n_excluded} "
        f"(+{n_excluded} excluded) train, {len(val_samples)} val"
    )

    for split, samples in [("train", train_samples), ("val", val_samples)]:
        for sample in samples:
            trial_id = sample["trial_id"]
            frame_idx = sample["frame_idx"]
            sample_id = _make_sample_id(trial_id, frame_idx)
            img_w, img_h = _get_image_dimensions(trial_id, frame_idx)

            src_path = _resolve_frame_path(trial_id, frame_idx)
            if src_path is None:
                continue
            ext = src_path.suffix
            dst_img = output_dir / split / "images" / f"{sample_id}{ext}"
            _copy_frame_image(trial_id, frame_idx, dst_img)

            label_lines = []
            for obj in sample["objs"]:
                x1, y1, x2, y2 = obj.bbox
                cx = _clip_and_normalize((x1 + x2) / 2, img_w)
                cy = _clip_and_normalize((y1 + y2) / 2, img_h)
                w = _clip_and_normalize(x2 - x1, img_w)
                h = _clip_and_normalize(y2 - y1, img_h)
                parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
                for kp_x, kp_y, kp_v in obj.keypoints:
                    if kp_v > 0:
                        nx = _clip_and_normalize(kp_x, img_w)
                        ny = _clip_and_normalize(kp_y, img_h)
                        parts.extend([f"{nx:.6f}", f"{ny:.6f}", str(kp_v)])
                    else:
                        parts.extend(["0.000000", "0.000000", "0"])
                label_lines.append(" ".join(parts))
            label_path = output_dir / split / "labels" / f"{sample_id}.txt"
            label_path.write_text("\n".join(label_lines), encoding="utf-8")

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {output_dir.absolute()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"task: pose\n"
        f"nc: 1\n"
        f"names:\n  0: peg\n"
        f"\n"
        f"kpt_shape: [3, 3]\n"
        f"flip_idx: {PEG_FLIP_IDX}\n",
        encoding="utf-8",
    )
    print(f"Peg pose dataset written to {output_dir}")
    return data_yaml


def export_peg_all(
    annotations_dir: Path = ANNOTATIONS_DIR,
    output_base_dir: Path = OUTPUT_BASE_DIR,
    train_split: float = 0.8,
    seed: int = 42,
) -> dict[str, Path]:
    """Export all peg dataset variants (segment, detect, pose).

    Returns:
        Dict mapping task name to data.yaml path.
    """
    results: dict[str, Path] = {}
    print("=" * 60)
    print("Exporting peg segmentation dataset...")
    print("=" * 60)
    results["peg_segment"] = export_peg_segmentation_dataset(
        annotations_dir, output_base_dir / "peg_segment", train_split, seed
    )
    print()
    print("=" * 60)
    print("Exporting peg detection dataset...")
    print("=" * 60)
    results["peg_detect"] = export_peg_detection_dataset(
        annotations_dir, output_base_dir / "peg_detect", train_split, seed
    )
    print()
    print("=" * 60)
    print("Exporting peg pose dataset...")
    print("=" * 60)
    results["peg_pose"] = export_peg_pose_dataset(
        annotations_dir, output_base_dir / "peg_pose", train_split, seed
    )
    print()
    print("=" * 60)
    print("All peg exports complete!")
    for task, yaml_path in results.items():
        print(f"  {task}: {yaml_path}")
    print("=" * 60)
    return results


# ==============================================================================
# Export All
# ==============================================================================


def export_all(
    annotations_dir: Path = ANNOTATIONS_DIR,
    output_base_dir: Path = OUTPUT_BASE_DIR,
    train_split: float = 0.8,
    seed: int = 42,
    include_negatives: bool = True,
    negative_ratio: float = 1.0,
) -> dict[str, Path]:
    """Export all dataset variants (segment, pose, pose_roi, detect).

    Args:
        annotations_dir: Directory with annotation JSONs.
        output_base_dir: Base output directory.
        train_split: Train/val split ratio.
        seed: Random seed.
        include_negatives: Whether to include negative (background) frames.
        negative_ratio: Max negatives as fraction of positives.

    Returns:
        Dict mapping task name to data.yaml path.
    """
    results = {}

    print("=" * 60)
    print("Exporting segmentation dataset...")
    print("=" * 60)
    results["segment"] = export_segmentation_dataset(
        annotations_dir, output_base_dir / "segment", train_split, seed,
        include_negatives, negative_ratio,
    )

    print()
    print("=" * 60)
    print("Exporting pose dataset (full images)...")
    print("=" * 60)
    results["pose"] = export_pose_dataset(
        annotations_dir, output_base_dir / "pose",
        use_roi_crops=False, train_split=train_split, seed=seed,
    )

    print()
    print("=" * 60)
    print("Exporting pose ROI dataset (cropped per-tool)...")
    print("=" * 60)
    results["pose_roi"] = export_pose_dataset(
        annotations_dir, output_base_dir / "pose_roi",
        use_roi_crops=True, train_split=train_split, seed=seed,
    )

    print()
    print("=" * 60)
    print("Exporting detection dataset...")
    print("=" * 60)
    results["detect"] = export_detection_dataset(
        annotations_dir, output_base_dir / "detect", train_split, seed,
        include_negatives, negative_ratio,
    )

    print()
    print("=" * 60)
    print("All exports complete!")
    for task, yaml_path in results.items():
        print(f"  {task}: {yaml_path}")
    print("=" * 60)

    return results


# ==============================================================================
# CLI
# ==============================================================================


def main() -> None:
    """CLI entry point for YOLO dataset export."""
    parser = argparse.ArgumentParser(
        description="Export annotation JSONs to YOLO-format datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m surgical_annotator.export_yolo --task all\n"
            "  python -m surgical_annotator.export_yolo --task segment\n"
            "  python -m surgical_annotator.export_yolo --task pose_roi\n"
            "  python -m surgical_annotator.export_yolo --task detect\n"
            "  python -m surgical_annotator.export_yolo --task peg_all\n"
            "  python -m surgical_annotator.export_yolo --task peg_segment\n"
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=[
            "all", "segment", "pose", "pose_roi", "detect",
            "peg_all", "peg_segment", "peg_detect", "peg_pose",
        ],
        help="Which dataset type to export (default: all). The peg_* tasks "
             "export the peg/pegboard datasets and are NOT included in 'all'.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=ANNOTATIONS_DIR,
        help=f"Annotations directory (default: {ANNOTATIONS_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_BASE_DIR,
        help=f"Output base directory (default: {OUTPUT_BASE_DIR}).",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split (default: 42).",
    )
    parser.add_argument(
        "--no-negatives",
        action="store_true",
        default=False,
        help="Skip automatic inclusion of negative (background) frames.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Max negatives as fraction of positives (default: 1.0).",
    )
    parser.add_argument(
        "--train-datasets",
        type=str,
        nargs="+",
        default=None,
        help="Dataset prefixes for train split (e.g. 7DOF2024). "
             "When specified, routes by trial_id prefix instead of random split.",
    )
    parser.add_argument(
        "--val-datasets",
        type=str,
        nargs="+",
        default=None,
        help="Dataset prefixes for val split (e.g. 6DOF2023). "
             "When specified, routes by trial_id prefix instead of random split.",
    )
    parser.add_argument(
        "--uncapped-neg-datasets",
        type=str,
        nargs="+",
        default=None,
        help="Dataset prefixes whose negatives bypass the ratio cap (all included).",
    )
    parser.add_argument(
        "--neg-ratio-override",
        type=str,
        nargs="+",
        default=None,
        help="Per-dataset negative ratio overrides as DATASET=RATIO "
             "(e.g. BAPES2024=3.0).",
    )

    args = parser.parse_args()

    include_neg = not args.no_negatives
    neg_ratio = args.negative_ratio

    uncapped_neg = set(args.uncapped_neg_datasets) if args.uncapped_neg_datasets else None
    ratio_overrides: dict[str, float] | None = None
    if args.neg_ratio_override:
        ratio_overrides = {}
        for item in args.neg_ratio_override:
            ds, ratio_str = item.split("=", 1)
            ratio_overrides[ds] = float(ratio_str)

    if args.task == "all":
        export_all(
            args.annotations_dir, args.output_dir, args.train_split, args.seed,
            include_neg, neg_ratio,
        )
    elif args.task == "segment":
        export_segmentation_dataset(
            args.annotations_dir, args.output_dir / "segment",
            args.train_split, args.seed, include_neg, neg_ratio,
            train_datasets=args.train_datasets,
            val_datasets=args.val_datasets,
            uncapped_neg_datasets=uncapped_neg,
            neg_ratio_overrides=ratio_overrides,
        )
    elif args.task == "pose":
        export_pose_dataset(
            args.annotations_dir, args.output_dir / "pose",
            use_roi_crops=False, train_split=args.train_split, seed=args.seed,
        )
    elif args.task == "pose_roi":
        export_pose_dataset(
            args.annotations_dir, args.output_dir / "pose_roi",
            use_roi_crops=True, train_split=args.train_split, seed=args.seed,
        )
    elif args.task == "detect":
        export_detection_dataset(
            args.annotations_dir, args.output_dir / "detect",
            args.train_split, args.seed, include_neg, neg_ratio,
            uncapped_neg_datasets=uncapped_neg,
            neg_ratio_overrides=ratio_overrides,
        )
    elif args.task == "peg_all":
        export_peg_all(
            args.annotations_dir, args.output_dir, args.train_split, args.seed,
        )
    elif args.task == "peg_segment":
        export_peg_segmentation_dataset(
            args.annotations_dir, args.output_dir / "peg_segment",
            args.train_split, args.seed,
        )
    elif args.task == "peg_detect":
        export_peg_detection_dataset(
            args.annotations_dir, args.output_dir / "peg_detect",
            args.train_split, args.seed,
        )
    elif args.task == "peg_pose":
        export_peg_pose_dataset(
            args.annotations_dir, args.output_dir / "peg_pose",
            args.train_split, args.seed,
        )


if __name__ == "__main__":
    main()
