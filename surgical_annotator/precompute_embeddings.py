"""Pre-compute embeddings from SAM2 and Depth Anything v2.

Extracts intermediate feature representations (before final output heads) from:
- SAM2: Image encoder embeddings from Hiera backbone
- Depth Anything v2: DINOv2 backbone features before depth head + depth images

These embeddings can be used for downstream tasks like:
- Transfer learning / fine-tuning
- Feature-based tool detection
- Representation learning

Output structure per trial:
  Trial1/
    DEPTH/
      frame_00000.npz      # Depth data (depth, depth_raw, min_depth, max_depth)
      frame_00000.png      # Colorized depth visualization
      embeddings/
        frame_00000.npz    # DINOv2 backbone features
    SAM/
      embeddings/
        frame_00000.npz    # SAM Hiera backbone features

Usage:
    python -m surgical_annotator.precompute_embeddings                    # sampled frames, all datasets
    python -m surgical_annotator.precompute_embeddings --all-frames       # ALL frames (not just sampled)
    python -m surgical_annotator.precompute_embeddings --dataset 7DOF2024 # one dataset
    python -m surgical_annotator.precompute_embeddings --trial Trial1     # specific trial
    python -m surgical_annotator.precompute_embeddings --resume           # skip existing
    python -m surgical_annotator.precompute_embeddings --sam-only         # only SAM embeddings
    python -m surgical_annotator.precompute_embeddings --depth-only       # only depth embeddings
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


def _load_sam_model():
    """Load SAM2 model for embedding extraction.

    Returns:
        Tuple of (predictor, device) or (None, None) if unavailable.
    """
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM2 model on {device}...", flush=True)

        predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2.1-hiera-small"
        )

        print("SAM2 model loaded.", flush=True)
        return predictor, device

    except ImportError as e:
        print(
            f"ERROR: SAM2 not installed. Install with:\n"
            f"  pip install sam2 torch torchvision\n"
            f"Details: {e}"
        )
        return None, None
    except Exception as e:
        print(f"ERROR: Failed to load SAM2: {e}")
        return None, None


def _load_depth_model():
    """Load Depth Anything v2 model for embedding extraction.

    Returns:
        Tuple of (model, processor, device) or (None, None, None) if unavailable.
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Depth Anything v2 Small on {device}...", flush=True)

        processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        ).to(device)
        model.eval()

        print("Depth Anything v2 Small loaded.", flush=True)
        return model, processor, device

    except ImportError as e:
        print(
            f"ERROR: Required packages not installed. Install with:\n"
            f"  pip install transformers torch\n"
            f"Details: {e}"
        )
        return None, None, None
    except Exception as e:
        print(f"ERROR: Failed to load Depth Anything v2: {e}")
        return None, None, None


def _discover_sampled_frames(
    fm: FrameManager,
    dataset_name: str,
    trial_filter: str | None = None,
) -> list[tuple[str, Path, list[Path]]]:
    """Discover sampled frames for trials in a dataset using FrameManager.

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


def _discover_all_trial_frames(
    fm: FrameManager,
    dataset_name: str,
    trial_filter: str | None = None,
) -> list[tuple[str, Path, list[Path]]]:
    """Discover ALL frames (not just sampled) for trials in a dataset.

    Args:
        fm: FrameManager instance with trials already discovered.
        dataset_name: Name of the dataset (e.g., "7DOF2024").
        trial_filter: Optional trial name filter (e.g., "Trial1").

    Returns:
        List of (trial_id, trial_path, frame_paths) tuples with ALL frames.
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

        # Glob ALL frames from frames_dir (not just sampled)
        frames_dir = trial_info.frames_dir
        frame_paths = []

        # Try different frame naming patterns
        for pattern in ["frame_*.bmp", "frame_*.png", "*.bmp", "*.png"]:
            frame_paths = sorted(frames_dir.glob(pattern))
            if frame_paths:
                break

        if frame_paths:
            results.append((trial_id, trial_info.trial_path, frame_paths))
            logger.debug(f"{trial_id}: Found {len(frame_paths)} total frames")

    return results


def _frame_index_from_path(frame_path: Path) -> int:
    """Extract frame index from a frame file path."""
    stem = frame_path.stem
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return int(digits) if digits else 0


def _extract_sam_embedding(
    predictor, device: str, image: np.ndarray
) -> np.ndarray | None:
    """Extract SAM2 image encoder embedding.

    SAM2 uses a Hiera backbone. After set_image(), the embedding is cached
    in predictor._features['image_embed'].

    Args:
        predictor: SAM2ImagePredictor instance.
        device: Device string.
        image: RGB image as numpy array (H, W, 3).

    Returns:
        Image embedding array or None on failure.
    """
    import torch

    try:
        # Set image to compute embedding
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(image)
        else:
            predictor.set_image(image)

        # Extract the cached image embedding
        # SAM2 stores features after encoding in _features dict
        if hasattr(predictor, '_features') and predictor._features is not None:
            img_embed = predictor._features.get('image_embed')
            if img_embed is not None:
                # Convert to numpy, handle potential batch dimension
                embed_np = img_embed.detach().cpu().float().numpy()
                if embed_np.ndim == 4 and embed_np.shape[0] == 1:
                    embed_np = embed_np.squeeze(0)  # Remove batch dim
                return embed_np

        # Alternative: try _image_embeddings attribute
        if hasattr(predictor, '_image_embeddings') and predictor._image_embeddings is not None:
            embed_np = predictor._image_embeddings.detach().cpu().float().numpy()
            if embed_np.ndim == 4 and embed_np.shape[0] == 1:
                embed_np = embed_np.squeeze(0)
            return embed_np

        print("  Warning: Could not extract SAM embedding (features not found)")
        return None

    except Exception as e:
        print(f"  Error extracting SAM embedding: {e}")
        return None


def _extract_depth_embedding(
    model, processor, device: str, image: Image.Image
) -> np.ndarray | None:
    """Extract Depth Anything v2 backbone embedding (before depth head).

    DA2 uses a DINOv2 backbone. We hook into the model to get features
    before the depth prediction head.

    Args:
        model: AutoModelForDepthEstimation instance.
        processor: HuggingFace image processor.
        device: Device string.
        image: PIL Image (RGB).

    Returns:
        Backbone embedding array or None on failure.
    """
    import torch

    try:
        # Process image
        inputs = processor(images=image, return_tensors="pt").to(device)

        # We need to get the intermediate features before the depth head
        # The model structure is: backbone -> neck -> head
        # We want backbone output

        features = None

        def hook_fn(module, input, output):
            nonlocal features
            features = output

        # Register hook on backbone (DINOv2)
        # The backbone is usually model.backbone or model.pretrained
        backbone = None
        if hasattr(model, 'backbone'):
            backbone = model.backbone
        elif hasattr(model, 'pretrained'):
            backbone = model.pretrained

        if backbone is None:
            # Try to get from config
            print("  Warning: Could not find backbone module")
            return None

        handle = backbone.register_forward_hook(hook_fn)

        try:
            with torch.inference_mode():
                _ = model(**inputs)
        finally:
            handle.remove()

        if features is None:
            print("  Warning: Hook did not capture features")
            return None

        # Convert to numpy - handle various output types
        if isinstance(features, torch.Tensor):
            embed_np = features.detach().cpu().float().numpy()
        elif isinstance(features, (list, tuple)):
            # Some backbones return multiple scales, take the last one
            embed_np = features[-1].detach().cpu().float().numpy()
        elif hasattr(features, 'feature_maps'):
            # BackboneOutput from transformers - use last feature map
            feat_maps = features.feature_maps
            if isinstance(feat_maps, (list, tuple)) and len(feat_maps) > 0:
                embed_np = feat_maps[-1].detach().cpu().float().numpy()
            else:
                print(f"  Warning: Empty feature_maps in BackboneOutput")
                return None
        elif hasattr(features, 'last_hidden_state'):
            # Some models return this attribute
            embed_np = features.last_hidden_state.detach().cpu().float().numpy()
        else:
            print(f"  Warning: Unexpected feature type: {type(features)}")
            return None

        # Remove batch dimension if present
        if embed_np.ndim == 4 and embed_np.shape[0] == 1:
            embed_np = embed_np.squeeze(0)
        elif embed_np.ndim == 3 and embed_np.shape[0] == 1:
            embed_np = embed_np.squeeze(0)

        return embed_np

    except Exception as e:
        print(f"  Error extracting depth embedding: {e}")
        return None


def _compute_depth_map(
    model, processor, device: str, image: Image.Image
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Compute depth map from image using Depth Anything v2.

    Args:
        model: AutoModelForDepthEstimation instance.
        processor: HuggingFace image processor.
        device: Device string.
        image: PIL Image (RGB).

    Returns:
        Tuple of (depth_normalized, depth_raw, min_depth, max_depth) or None on failure.
        - depth_normalized: Normalized 0-1 depth at original image size (H, W)
        - depth_raw: Raw model output at model resolution
        - min_depth, max_depth: For denormalization
    """
    import torch

    try:
        original_size = image.size  # (W, H)

        # Process image for model
        inputs = processor(images=image, return_tensors="pt").to(device)

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

        return depth_normalized.astype(np.float32), depth_raw_np.astype(np.float32), min_depth, max_depth

    except Exception as e:
        print(f"  Error computing depth map: {e}")
        return None


def precompute_trial_embeddings(
    sam_predictor,
    sam_device: str | None,
    depth_model,
    depth_processor,
    depth_device: str | None,
    trial_id: str,
    trial_path: Path,
    frame_paths: list[Path],
    resume: bool = False,
    sam_only: bool = False,
    depth_only: bool = False,
) -> dict[str, int]:
    """Pre-compute embeddings and depth maps for frames in a trial.

    Outputs per frame:
    - SAM/embeddings/frame_XXXXX.npz: SAM Hiera backbone features
    - DEPTH/frame_XXXXX.npz: Depth data (depth, depth_raw, min_depth, max_depth)
    - DEPTH/frame_XXXXX.png: Colorized depth visualization
    - DEPTH/embeddings/frame_XXXXX.npz: DINOv2 backbone features

    Args:
        sam_predictor: SAM2ImagePredictor or None.
        sam_device: Device for SAM or None.
        depth_model: Depth model or None.
        depth_processor: Depth processor or None.
        depth_device: Device for depth or None.
        trial_id: Trial identifier.
        trial_path: Path to trial directory.
        frame_paths: List of frame image paths.
        resume: If True, skip existing files.
        sam_only: Only extract SAM embeddings.
        depth_only: Only extract depth embeddings (includes depth images).

    Returns:
        Dict with counts: {sam_embeddings, depth_embeddings, depth_images}
    """
    # Create output directories
    sam_embed_dir = trial_path / "SAM" / "embeddings"
    depth_dir = trial_path / "DEPTH"
    depth_embed_dir = depth_dir / "embeddings"

    if sam_predictor and not depth_only:
        sam_embed_dir.mkdir(parents=True, exist_ok=True)
    if depth_model and not sam_only:
        depth_dir.mkdir(parents=True, exist_ok=True)
        depth_embed_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "sam_embeddings": 0,
        "depth_embeddings": 0,
        "depth_images": 0,
    }

    for i, frame_path in enumerate(frame_paths):
        frame_idx = _frame_index_from_path(frame_path)

        # ===== SAM Embedding =====
        if sam_predictor and not depth_only:
            sam_npz = sam_embed_dir / f"frame_{frame_idx:05d}.npz"
            if not (resume and sam_npz.exists()):
                try:
                    # Load as numpy array for SAM
                    image_np = np.array(Image.open(frame_path).convert("RGB"))

                    embedding = _extract_sam_embedding(
                        sam_predictor, sam_device, image_np
                    )
                    if embedding is not None:
                        np.savez_compressed(
                            sam_npz,
                            embedding=embedding,
                            shape=embedding.shape,
                        )
                        counts["sam_embeddings"] += 1
                except Exception as e:
                    print(f"  ERROR SAM frame {frame_idx}: {e}")

        # ===== Depth Processing =====
        if depth_model and not sam_only:
            depth_npz = depth_dir / f"frame_{frame_idx:05d}.npz"
            depth_png = depth_dir / f"frame_{frame_idx:05d}.png"
            embed_npz = depth_embed_dir / f"frame_{frame_idx:05d}.npz"

            # Check what needs to be computed
            depth_image_exists = depth_npz.exists() and depth_png.exists()
            depth_embed_exists = embed_npz.exists()

            # Load image once for all depth operations
            image_pil = None
            if not (resume and depth_image_exists and depth_embed_exists):
                image_pil = Image.open(frame_path).convert("RGB")

            # --- Depth Images (NPZ + PNG) ---
            if not (resume and depth_image_exists):
                try:
                    if image_pil is None:
                        image_pil = Image.open(frame_path).convert("RGB")

                    result = _compute_depth_map(
                        depth_model, depth_processor, depth_device, image_pil
                    )
                    if result is not None:
                        depth_norm, depth_raw, min_depth, max_depth = result

                        # Save NPZ with depth data
                        np.savez_compressed(
                            depth_npz,
                            depth=depth_norm,
                            depth_raw=depth_raw,
                            min_depth=min_depth,
                            max_depth=max_depth,
                        )

                        # Save colorized visualization
                        _save_depth_visualization(depth_norm, depth_png)
                        counts["depth_images"] += 1

                except Exception as e:
                    print(f"  ERROR Depth image frame {frame_idx}: {e}")

            # --- Depth Backbone Embedding ---
            if not (resume and depth_embed_exists):
                try:
                    if image_pil is None:
                        image_pil = Image.open(frame_path).convert("RGB")

                    embedding = _extract_depth_embedding(
                        depth_model, depth_processor, depth_device, image_pil
                    )
                    if embedding is not None:
                        np.savez_compressed(
                            embed_npz,
                            embedding=embedding,
                            shape=embedding.shape,
                        )
                        counts["depth_embeddings"] += 1
                except Exception as e:
                    print(f"  ERROR Depth embedding frame {frame_idx}: {e}")

        # Progress update
        if (i + 1) % 50 == 0 or i == len(frame_paths) - 1:
            print(f"  {trial_id}: {i + 1}/{len(frame_paths)} frames", flush=True)

    return counts


def main():
    # Force unbuffered output for progress visibility
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Pre-compute SAM and Depth embeddings (+ depth images) for frames"
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
        "--all-frames",
        action="store_true",
        help="Process ALL frames (not just sampled annotation frames)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip frames that already have output files",
    )
    parser.add_argument(
        "--sam-only",
        action="store_true",
        help="Only extract SAM embeddings",
    )
    parser.add_argument(
        "--depth-only",
        action="store_true",
        help="Only extract Depth Anything outputs (embeddings + depth images)",
    )
    args = parser.parse_args()

    if args.sam_only and args.depth_only:
        print("ERROR: Cannot specify both --sam-only and --depth-only")
        sys.exit(1)

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

    # Initialize FrameManager
    fm = FrameManager()
    fm.discover_all_trials()

    # Load models
    sam_predictor, sam_device = None, None
    depth_model, depth_processor, depth_device = None, None, None

    if not args.depth_only:
        sam_predictor, sam_device = _load_sam_model()
        if sam_predictor is None and not args.sam_only:
            print("Warning: SAM model not available, skipping SAM embeddings")
        elif sam_predictor is None and args.sam_only:
            print("ERROR: --sam-only specified but SAM model not available")
            sys.exit(1)

    if not args.sam_only:
        depth_model, depth_processor, depth_device = _load_depth_model()
        if depth_model is None and not args.depth_only:
            print("Warning: Depth model not available, skipping depth embeddings")
        elif depth_model is None and args.depth_only:
            print("ERROR: --depth-only specified but Depth model not available")
            sys.exit(1)

    if sam_predictor is None and depth_model is None:
        print("ERROR: No models available. Nothing to do.")
        sys.exit(1)

    totals = {
        "sam_embeddings": 0,
        "depth_embeddings": 0,
        "depth_images": 0,
    }
    start_time = time.time()

    # Select frame discovery mode
    frame_mode = "all" if args.all_frames else "sampled"

    for dataset_name in datasets_ordered:
        print(f"\n{'='*60}", flush=True)
        print(f"Dataset: {dataset_name}", flush=True)
        print(f"Mode: {frame_mode} frames", flush=True)
        print(f"{'='*60}", flush=True)

        # Discover frames based on mode
        if args.all_frames:
            trials = _discover_all_trial_frames(fm, dataset_name, args.trial)
        else:
            trials = _discover_sampled_frames(fm, dataset_name, args.trial)

        print(f"Found {len(trials)} trials", flush=True)

        for trial_id, trial_path, frame_paths in trials:
            print(f"\n  {trial_id}: {len(frame_paths)} {frame_mode} frames", flush=True)

            counts = precompute_trial_embeddings(
                sam_predictor=sam_predictor,
                sam_device=sam_device,
                depth_model=depth_model,
                depth_processor=depth_processor,
                depth_device=depth_device,
                trial_id=trial_id,
                trial_path=trial_path,
                frame_paths=frame_paths,
                resume=args.resume,
                sam_only=args.sam_only,
                depth_only=args.depth_only,
            )

            # Accumulate totals
            for key in totals:
                totals[key] += counts.get(key, 0)

            # Report per-trial status
            status_parts = []
            if counts.get("sam_embeddings", 0) > 0:
                status_parts.append(f"SAM: {counts['sam_embeddings']}")
            if counts.get("depth_images", 0) > 0:
                status_parts.append(f"Depth imgs: {counts['depth_images']}")
            if counts.get("depth_embeddings", 0) > 0:
                status_parts.append(f"Depth embed: {counts['depth_embeddings']}")

            if status_parts:
                print(f"  -> Processed: {', '.join(status_parts)}", flush=True)
            else:
                print(f"  -> All outputs already computed (skipped)", flush=True)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}", flush=True)
    print(f"Done in {elapsed:.1f}s ({frame_mode} frames)", flush=True)
    if not args.depth_only:
        print(f"  SAM embeddings:   {totals['sam_embeddings']}", flush=True)
    if not args.sam_only:
        print(f"  Depth images:     {totals['depth_images']}", flush=True)
        print(f"  Depth embeddings: {totals['depth_embeddings']}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
