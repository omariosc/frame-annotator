"""SAM2 segmentation integration for AI-assisted annotation.

Uses Meta's Segment Anything Model 2 (SAM2) to generate precise
segmentation masks from point prompts, improving annotation speed
and accuracy for surgical tool segmentation.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for SAM2 (heavy dependencies)
_sam_predictor = None
_sam_device = None
_cached_image_hash = None


def _get_device() -> str:
    """Determine best available device for SAM inference."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_predictor():
    """Load SAM2 model predictor (lazy, singleton).

    Returns:
        SAM2ImagePredictor instance or None if SAM2 not available.
    """
    global _sam_predictor, _sam_device

    if _sam_predictor is not None:
        return _sam_predictor

    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        _sam_device = _get_device()
        logger.info(f"Loading SAM2 model on {_sam_device}...")

        # Use the small model for good balance of speed and quality
        _sam_predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2.1-hiera-small"
        )

        logger.info("SAM2 model loaded successfully.")
        return _sam_predictor

    except ImportError:
        logger.error(
            "SAM2 not installed. Install with: pip install sam2\n"
            "Also requires: pip install torch torchvision"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to load SAM2 model: {e}")
        return None


def _image_hash(image: np.ndarray) -> int:
    """Compute a fast hash of an image for caching."""
    # Use a strided sample for speed
    sample = image[::16, ::16].tobytes()
    return hash(sample)


def set_image(image: np.ndarray) -> bool:
    """Set the current image for SAM inference (caches embedding).

    Args:
        image: RGB image as numpy array (H, W, 3), uint8.

    Returns:
        True if image was set successfully.
    """
    global _cached_image_hash

    predictor = _load_predictor()
    if predictor is None:
        return False

    img_hash = _image_hash(image)
    if img_hash == _cached_image_hash:
        return True  # Already cached

    try:
        import torch
        with torch.inference_mode():
            if _sam_device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predictor.set_image(image)
            else:
                predictor.set_image(image)
        _cached_image_hash = img_hash
        return True
    except Exception as e:
        logger.error(f"Failed to set image for SAM: {e}")
        _cached_image_hash = None
        return False


def segment_point(
    point_coords: list[list[float]],
    point_labels: list[int],
    multimask: bool = True,
) -> Optional[dict]:
    """Run SAM segmentation with point prompts.

    Args:
        point_coords: List of [x, y] pixel coordinates.
        point_labels: List of labels (1=foreground, 0=background).
        multimask: If True, returns 3 mask proposals; else returns best one.

    Returns:
        Dict with 'masks' (list of polygon vertex lists), 'scores' (IoU scores),
        and 'best_idx' (index of highest-scoring mask), or None on failure.
    """
    predictor = _load_predictor()
    if predictor is None:
        return None

    try:
        import torch

        coords = np.array(point_coords, dtype=np.float32)
        labels = np.array(point_labels, dtype=np.int32)

        with torch.inference_mode():
            if _sam_device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, _ = predictor.predict(
                        point_coords=coords,
                        point_labels=labels,
                        multimask_output=multimask,
                    )
            else:
                masks, scores, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=multimask,
                )

        # Convert binary masks to polygons
        polygons = []
        for mask in masks:
            poly = _mask_to_polygon(mask)
            polygons.append(poly)

        best_idx = int(np.argmax(scores))

        return {
            "masks": polygons,
            "scores": [float(s) for s in scores],
            "best_idx": best_idx,
        }

    except Exception as e:
        logger.error(f"SAM segmentation failed: {e}")
        return None


def segment_box(
    box: list[float],
    point_coords: Optional[list[list[float]]] = None,
    point_labels: Optional[list[int]] = None,
) -> Optional[dict]:
    """Run SAM segmentation with a bounding box prompt.

    Args:
        box: [x_min, y_min, x_max, y_max] bounding box.
        point_coords: Optional additional point prompts.
        point_labels: Optional labels for points.

    Returns:
        Dict with 'masks', 'scores', 'best_idx', or None on failure.
    """
    predictor = _load_predictor()
    if predictor is None:
        return None

    try:
        import torch

        box_arr = np.array(box, dtype=np.float32)
        coords = np.array(point_coords, dtype=np.float32) if point_coords else None
        labels = np.array(point_labels, dtype=np.int32) if point_labels else None

        with torch.inference_mode():
            if _sam_device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, _ = predictor.predict(
                        point_coords=coords,
                        point_labels=labels,
                        box=box_arr,
                        multimask_output=False,
                    )
            else:
                masks, scores, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    box=box_arr,
                    multimask_output=False,
                )

        polygons = []
        for mask in masks:
            poly = _mask_to_polygon(mask)
            polygons.append(poly)

        best_idx = int(np.argmax(scores))

        return {
            "masks": polygons,
            "scores": [float(s) for s in scores],
            "best_idx": best_idx,
        }

    except Exception as e:
        logger.error(f"SAM box segmentation failed: {e}")
        return None


def _mask_to_polygon(
    mask: np.ndarray,
    tolerance: float = 2.0,
    min_area: int = 100,
) -> list[list[float]]:
    """Convert a binary mask to polygon vertices.

    Uses contour detection and Douglas-Peucker simplification
    to produce a clean polygon suitable for annotation.

    Args:
        mask: Binary mask (H, W), boolean or uint8.
        tolerance: Douglas-Peucker simplification tolerance in pixels.
        min_area: Minimum contour area to consider.

    Returns:
        List of [x, y] vertex coordinates for the largest contour.
    """
    try:
        import cv2

        mask_uint8 = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        # Find largest contour by area
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) < min_area:
            return []

        # Simplify with Douglas-Peucker
        epsilon = tolerance
        simplified = cv2.approxPolyDP(largest, epsilon, True)

        # Convert to list of [x, y]
        vertices = simplified.squeeze().tolist()

        # Ensure it's a list of lists (not a single point)
        if isinstance(vertices[0], (int, float)):
            vertices = [vertices]

        return [[float(v[0]), float(v[1])] for v in vertices]

    except ImportError:
        # Fallback without OpenCV: use skimage
        try:
            from skimage.measure import find_contours

            contours = find_contours(mask.astype(float), 0.5)
            if not contours:
                return []

            # Largest contour
            largest = max(contours, key=len)

            # Subsample if too many points
            if len(largest) > 200:
                step = max(1, len(largest) // 100)
                largest = largest[::step]

            # find_contours returns (row, col) = (y, x)
            return [[float(pt[1]), float(pt[0])] for pt in largest]

        except ImportError:
            logger.error("Neither cv2 nor skimage available for mask-to-polygon")
            return []


_auto_generator = None


def get_auto_generator():
    """Lazy-load SAM2AutomaticMaskGenerator singleton.

    Uses the same model and config as precompute_sam.py for consistency.

    Returns:
        SAM2AutomaticMaskGenerator instance or None if unavailable.
    """
    global _auto_generator

    if _auto_generator is not None:
        return _auto_generator

    try:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = _get_device()
        logger.info(f"Loading SAM2 automatic mask generator on {device}...")

        predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2.1-hiera-small"
        )

        _auto_generator = SAM2AutomaticMaskGenerator(
            model=predictor.model,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            min_mask_region_area=100,
        )

        logger.info("SAM2 automatic mask generator loaded.")
        return _auto_generator

    except ImportError:
        logger.error(
            "SAM2 not installed. Install with: pip install sam2\n"
            "Also requires: pip install torch torchvision"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to load SAM2 automatic mask generator: {e}")
        return None


def generate_automatic_masks(image: np.ndarray) -> list[dict]:
    """Run automatic mask generation on an image.

    Args:
        image: RGB image as numpy array (H, W, 3), uint8.

    Returns:
        List of mask dicts from SAM (each has 'segmentation', 'area',
        'bbox', 'predicted_iou', 'stability_score'), or empty list on
        failure.
    """
    gen = get_auto_generator()
    if gen is None:
        return []

    try:
        import torch
        with torch.inference_mode():
            return gen.generate(image)
    except Exception as e:
        logger.error(f"Automatic mask generation failed: {e}")
        return []


def is_available() -> bool:
    """Check if SAM2 is available without loading the model.

    Returns:
        True if SAM2 can be imported.
    """
    try:
        import sam2  # noqa: F401
        return True
    except ImportError:
        return False


def get_status() -> dict:
    """Get SAM2 status information.

    Returns:
        Dict with availability, model info, and device.
    """
    available = is_available()
    loaded = _sam_predictor is not None

    status = {
        "available": available,
        "loaded": loaded,
        "device": _sam_device if loaded else None,
        "model": "sam2.1-hiera-small" if loaded else None,
    }

    if not available:
        status["install_hint"] = "pip install sam2 torch torchvision"

    return status
