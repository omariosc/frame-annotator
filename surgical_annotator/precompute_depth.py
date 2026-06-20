"""Pre-compute depth maps and embeddings for sampled annotation frames.

Uses Depth Anything v2 Small (25M params) for fast inference with good quality.
Only processes frames that the annotation tool actually shows (sampled every 25
frames, skipping NaN kinematics), not all frames.

Saves per frame:
  - .npz data files (depth array, raw depth, min/max for denormalization) in DEPTH/
  - Colorized depth visualization images in DEPTH/
  - Multi-scale embeddings (global + ROI) in DEPTH/embeddings/ (optional)

Embeddings:
  - Global: Full-frame 384D embedding from DA2 DINOv2-S backbone
  - ROI: Per-tool 384D embeddings (if bounding boxes available)

Usage:
    python -m surgical_annotator.precompute_depth                    # all datasets
    python -m surgical_annotator.precompute_depth --dataset 7DOF2024 # one dataset
    python -m surgical_annotator.precompute_depth --trial Trial1     # specific trial
    python -m surgical_annotator.precompute_depth --resume           # skip existing
    python -m surgical_annotator.precompute_depth --embeddings       # extract embeddings
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
from surgical_annotator.frame_manager import FrameManager

logger = logging.getLogger(__name__)


def _load_depth_model(extract_embeddings: bool = False):
    """Load Depth Anything v2 Small model.

    Args:
        extract_embeddings: If True, also prepare for backbone feature extraction.

    Returns:
        Tuple of (model, processor, device, backbone) or (None, None, None, None).
        backbone is only set if extract_embeddings=True.
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Depth Anything v2 Small on {device}...")

        processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        ).to(device)
        model.eval()

        backbone = None
        if extract_embeddings:
            # The DA2 model uses a DINOv2-S backbone
            # Access it via model.backbone.embeddings and model.backbone.encoder
            # We'll use the pretrained backbone directly for embeddings
            backbone = model.backbone
            print("Depth Anything v2 backbone prepared for embedding extraction.")

        print("Depth Anything v2 Small loaded.")
        return model, processor, device, backbone

    except ImportError as e:
        print(
            f"ERROR: Required packages not installed. Install with:\n"
            f"  pip install transformers torch\n"
            f"Details: {e}"
        )
        return None, None, None, None
    except Exception as e:
        print(f"ERROR: Failed to load Depth Anything v2: {e}")
        return None, None, None, None


def _extract_backbone_embedding(
    model,
    image: "Image.Image",
    processor,
    device: str,
) -> np.ndarray:
    """Extract backbone embedding from image using DA2's DINOv2-S backbone.

    Args:
        model: Depth Anything v2 model with backbone attribute.
        image: PIL Image (RGB).
        processor: HuggingFace image processor.
        device: Device string.

    Returns:
        Embedding array of shape (384,) for DINOv2-S.
    """
    import torch

    # Process image
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.inference_mode():
        # Get backbone features (DINOv2-S outputs 384D)
        # The backbone is a DPT backbone wrapping DINOv2
        # We need to access the underlying encoder
        backbone = model.backbone

        # Get patch embeddings through the backbone
        # The HF model exposes backbone.embeddings and backbone.encoder
        pixel_values = inputs["pixel_values"]

        # Forward through embedding layer
        embeddings = backbone.embeddings(pixel_values)

        # Forward through encoder layers
        hidden_states = embeddings
        for layer in backbone.encoder.layer:
            hidden_states = layer(hidden_states)[0]

        # Extract CLS token (global representation)
        cls_token = hidden_states[:, 0, :]  # (B, 384)

        embedding = cls_token.squeeze(0).cpu().numpy()

    return embedding.astype(np.float32)


def _crop_roi(image: "Image.Image", bbox: list[float]) -> "Image.Image":
    """Crop image to bounding box and resize to standard size.

    Args:
        image: PIL Image.
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates.

    Returns:
        Cropped and resized PIL Image (224x224).
    """
    x1, y1, x2, y2 = [int(c) for c in bbox]

    # Ensure valid coordinates
    w, h = image.size
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    # Crop and resize
    cropped = image.crop((x1, y1, x2, y2))
    resized = cropped.resize((224, 224), Image.Resampling.BILINEAR)

    return resized


def _discover_sampled_frames(
    fm: FrameManager,
    dataset_name: str,
    trial_filter: str | None = None,
) -> list[tuple[str, Path, list[Path]]]:
    """Discover sampled frames for trials in a dataset using FrameManager.

    Only returns the frames the annotation tool actually shows (sampled
    every 25 frames, skipping NaN kinematics), not all frames.

    Args:
        fm: FrameManager instance with trials already discovered.
        dataset_name: Name of the dataset (e.g., "7DOF2024").
        trial_filter: Optional trial name filter (e.g., "Trial1").

    Returns:
        List of (trial_id, trial_path, frame_paths) tuples.
    """
    results = []
    all_trials = fm.discover_all_trials()
    trial_names = all_trials.get(dataset_name, [])

    for trial_name in trial_names:
        # Apply trial filter if specified
        if trial_filter and trial_filter not in trial_name:
            continue

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


def _save_depth_visualization(depth: np.ndarray, save_path: Path) -> None:
    """Save colorized depth map using turbo colormap.

    Args:
        depth: 2D depth array (H, W).
        save_path: Path to save the PNG.
    """
    import matplotlib.pyplot as plt

    # Normalize to 0-1
    depth_min = depth.min()
    depth_max = depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)

    # Apply turbo colormap (perceptually uniform, good for depth)
    colored = plt.cm.turbo(depth_norm)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)

    Image.fromarray(colored).save(save_path)


def precompute_trial(
    model,
    processor,
    device: str,
    trial_id: str,
    trial_path: Path,
    frame_paths: list[Path],
    resume: bool = False,
    extract_embeddings: bool = False,
    annotations: dict | None = None,
) -> int:
    """Pre-compute depth maps and embeddings for all sampled frames in a trial.

    Saves .npz data, colorized PNG visualization, and optionally embeddings.

    Args:
        model: Depth Anything v2 model.
        processor: HuggingFace image processor.
        device: Device string ("cuda" or "cpu").
        trial_id: Trial identifier string.
        trial_path: Path to trial directory.
        frame_paths: List of frame image paths.
        resume: If True, skip frames that already have .npz files.
        extract_embeddings: If True, also extract and save backbone embeddings.
        annotations: Optional dict mapping frame_idx to annotation with tool bboxes.

    Returns:
        Number of frames processed.
    """
    import torch

    # Create DEPTH directory
    depth_dir = trial_path / "DEPTH"
    depth_dir.mkdir(exist_ok=True)

    # Create embeddings subdirectory if extracting
    embed_dir = None
    if extract_embeddings:
        embed_dir = depth_dir / "embeddings"
        embed_dir.mkdir(exist_ok=True)

    processed = 0

    for i, frame_path in enumerate(frame_paths):
        frame_idx = _frame_index_from_path(frame_path)
        npz_path = depth_dir / f"frame_{frame_idx:05d}.npz"
        png_path = depth_dir / f"frame_{frame_idx:05d}.png"
        embed_path = embed_dir / f"frame_{frame_idx:05d}.npz" if embed_dir else None

        # Check what needs to be computed
        depth_exists = npz_path.exists()
        embed_exists = embed_path.exists() if embed_path else True

        if resume and depth_exists and embed_exists:
            continue

        try:
            # Load image
            image = Image.open(frame_path).convert("RGB")
            original_size = image.size  # (W, H)

            # Process image for model
            inputs = processor(images=image, return_tensors="pt").to(device)

            # Compute depth map if needed
            if not depth_exists or not resume:
                with torch.inference_mode():
                    outputs = model(**inputs)
                    depth_raw = outputs.predicted_depth

                # Interpolate to original image size
                depth = torch.nn.functional.interpolate(
                    depth_raw.unsqueeze(1),
                    size=(original_size[1], original_size[0]),  # (H, W)
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()

                # Get raw depth for storage
                depth_raw_np = depth_raw.squeeze().cpu().numpy()

                # Compute min/max for denormalization
                min_depth = float(depth.min())
                max_depth = float(depth.max())

                # Normalize depth to 0-1 range for storage
                depth_normalized = (depth - min_depth) / (max_depth - min_depth + 1e-8)

                # Save NPZ with depth data
                np.savez_compressed(
                    npz_path,
                    depth=depth_normalized.astype(np.float32),  # Normalized 0-1
                    depth_raw=depth_raw_np.astype(np.float32),  # Raw model output
                    min_depth=min_depth,
                    max_depth=max_depth,
                )

                # Save colorized visualization
                _save_depth_visualization(depth, png_path)

            # Extract embeddings if requested
            if extract_embeddings and embed_path and (not embed_exists or not resume):
                # Global embedding (full frame)
                global_embed = _extract_backbone_embedding(
                    model, image, processor, device
                )

                # ROI embeddings (per tool, if annotations available)
                roi1_embed = None
                roi2_embed = None

                if annotations and frame_idx in annotations:
                    ann = annotations[frame_idx]
                    # Try to get tool bboxes
                    tool1_bbox = None
                    tool2_bbox = None

                    if "tool1" in ann and "bbox" in ann["tool1"]:
                        tool1_bbox = ann["tool1"]["bbox"]
                    elif "tool1_bbox" in ann:
                        tool1_bbox = ann["tool1_bbox"]

                    if "tool2" in ann and "bbox" in ann["tool2"]:
                        tool2_bbox = ann["tool2"]["bbox"]
                    elif "tool2_bbox" in ann:
                        tool2_bbox = ann["tool2_bbox"]

                    if tool1_bbox:
                        roi1_image = _crop_roi(image, tool1_bbox)
                        roi1_embed = _extract_backbone_embedding(
                            model, roi1_image, processor, device
                        )

                    if tool2_bbox:
                        roi2_image = _crop_roi(image, tool2_bbox)
                        roi2_embed = _extract_backbone_embedding(
                            model, roi2_image, processor, device
                        )

                # Save embeddings
                embed_data = {"global_embedding": global_embed}
                if roi1_embed is not None:
                    embed_data["roi1_embedding"] = roi1_embed
                if roi2_embed is not None:
                    embed_data["roi2_embedding"] = roi2_embed

                np.savez_compressed(embed_path, **embed_data)

            processed += 1

        except Exception as e:
            print(f"  ERROR frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if (i + 1) % 50 == 0 or i == len(frame_paths) - 1:
            print(f"  {trial_id}: {i + 1}/{len(frame_paths)} frames")

    return processed


def _load_trial_annotations(
    annotation_dir: Path,
    dataset: str,
    trial_name: str,
) -> dict | None:
    """Load annotations for a trial.

    Args:
        annotation_dir: Directory with annotation files.
        dataset: Dataset name.
        trial_name: Trial name.

    Returns:
        Dict mapping frame_idx to annotation, or None if not found.
    """
    import json

    # Try different naming patterns
    patterns = [
        f"{dataset}_{trial_name}.json",
        f"{trial_name}.json",
        f"annotations_{trial_name}.json",
    ]

    for pattern in patterns:
        ann_file = annotation_dir / pattern
        if ann_file.exists():
            try:
                with open(ann_file) as f:
                    data = json.load(f)

                # Index by frame number
                if isinstance(data, list):
                    return {
                        item.get("frame_idx", item.get("frame", i)): item
                        for i, item in enumerate(data)
                    }
                elif isinstance(data, dict) and "frames" in data:
                    return {f["frame_idx"]: f for f in data["frames"]}
                else:
                    return data
            except Exception as e:
                print(f"  Warning: Failed to load annotations from {ann_file}: {e}")

    return None


def _discover_all_frames(
    fm: FrameManager,
    dataset_name: str,
    trial_filter: str | None = None,
) -> list[tuple[str, Path, list[Path]]]:
    """Discover ALL frames for trials in a dataset (not just sampled).

    Unlike _discover_sampled_frames which only returns annotation-sampled
    frames (every 100th), this returns every frame file in the Frames/
    directory. Used for temporal training where all frames in a window
    need precomputed visual features.

    Args:
        fm: FrameManager instance with trials already discovered.
        dataset_name: Name of the dataset (e.g., "7DOF2024").
        trial_filter: Optional trial name filter (e.g., "Trial1").

    Returns:
        List of (trial_id, trial_path, frame_paths) tuples.
    """
    results = []
    all_trials = fm.discover_all_trials()
    trial_names = all_trials.get(dataset_name, [])

    for trial_name in trial_names:
        if trial_filter and trial_filter not in trial_name:
            continue

        trial_id = f"{dataset_name}/{trial_name}"
        trial_info = fm.get_trial_info(trial_id)
        if not trial_info:
            continue

        frames_dir = trial_info.frames_dir
        if not frames_dir.is_dir():
            continue

        # Discover all frame files
        frame_paths = sorted(frames_dir.glob("frame_*.bmp"))
        if not frame_paths:
            frame_paths = sorted(frames_dir.glob("frame_*.png"))
        if not frame_paths:
            # Try the 6DOF pattern
            frame_paths = sorted(frames_dir.glob("*.png"))
            if not frame_paths:
                frame_paths = sorted(frames_dir.glob("*.bmp"))

        if frame_paths:
            results.append((trial_id, trial_info.trial_path, frame_paths))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute depth maps and embeddings for sampled annotation frames"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Process only this dataset (e.g. 7DOF2024)",
    )
    parser.add_argument(
        "--trial",
        type=str,
        default=None,
        help="Process only trials matching this name (e.g. Trial1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip frames that already have .npz files",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Extract backbone embeddings (global + ROI) for multi-modal training",
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Only extract embeddings (skip depth maps if they exist)",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Process ALL frames (not just annotation-sampled). "
             "Required for temporal training windows.",
    )
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

    # Load depth model
    extract_embeddings = args.embeddings or args.embeddings_only
    model, processor, device, backbone = _load_depth_model(
        extract_embeddings=extract_embeddings
    )
    if model is None:
        sys.exit(1)

    # Annotation directory for tool bboxes
    annotation_dir = Path("D:/Data/AI-ELT/outputs/annotations")

    total_processed = 0
    start_time = time.time()

    for dataset_name in datasets_ordered:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        if extract_embeddings:
            print(f"Mode: Depth maps + Embeddings")
        else:
            print(f"Mode: Depth maps only")
        print(f"{'='*60}")

        if args.all_frames:
            trials = _discover_all_frames(fm, dataset_name, args.trial)
            frame_mode = "ALL"
        else:
            trials = _discover_sampled_frames(fm, dataset_name, args.trial)
            frame_mode = "sampled"
        print(f"Found {len(trials)} trials ({frame_mode} frames)")

        for trial_id, trial_path, frame_paths in trials:
            print(f"\n  {trial_id}: {len(frame_paths)} {frame_mode} frames")

            # Load annotations for ROI embeddings
            annotations = None
            if extract_embeddings:
                trial_name = trial_path.name
                annotations = _load_trial_annotations(
                    annotation_dir, dataset_name, trial_name
                )
                if annotations:
                    print(f"  -> Found {len(annotations)} frame annotations for ROI extraction")

            count = precompute_trial(
                model,
                processor,
                device,
                trial_id,
                trial_path,
                frame_paths,
                resume=args.resume,
                extract_embeddings=extract_embeddings,
                annotations=annotations,
            )
            total_processed += count
            if count > 0:
                print(f"  -> Processed {count} frames")
            else:
                print(f"  -> All frames already computed (skipped)")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    mode_str = "all" if args.all_frames else "sampled"
    print(f"Done. Processed {total_processed} {mode_str} frames in {elapsed:.1f}s")
    if extract_embeddings:
        print(f"Embeddings saved to DEPTH/embeddings/ subdirectories")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
