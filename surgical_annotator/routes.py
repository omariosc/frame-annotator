"""API routes for annotation tool."""

import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
from io import BytesIO
from flask import Blueprint, jsonify, request, send_file, abort, Response
from PIL import Image

from .frame_manager import get_frame_manager
from .annotation_store import get_annotation_store, FrameAnnotation, ANNOTATION_DIR
from . import sam_segmentation
from .phase_definitions import get_definitions_dict

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

def _invalidate_progress_cache() -> None:
    """No-op: progress is now read from per-trial _progress.json files.

    Kept as a function so existing callers don't break. The summaries are
    updated atomically by AnnotationStore on each save.
    """
    pass


@api_bp.route('/datasets', methods=['GET'])
def get_datasets():
    """Get list of all datasets and their trials."""
    fm = get_frame_manager()
    datasets = fm.discover_all_trials()

    result = []
    for dataset_name, trials in datasets.items():
        result.append({
            'name': dataset_name,
            'trials': trials
        })

    return jsonify(result)


@api_bp.route('/trials/<path:trial_id>/frames', methods=['GET'])
def get_trial_frames(trial_id: str):
    """Get sampled frames for a trial.

    Args:
        trial_id: Trial identifier (e.g., "7DOF2024/Trial1")
    """
    fm = get_frame_manager()
    store = get_annotation_store()

    trial = fm.get_trial_info(trial_id)
    if not trial:
        logger.error(f"Trial not found: '{trial_id}'. Known trials: {len(fm.trials)}")
        abort(404, description=f"Trial not found: {trial_id}")

    # Get sampled frames with NaN filtering
    valid_frames = fm.get_sampled_frames(trial_id)

    # Load only sampled frames (not ALL 600+ files)
    store.load_trial_sampled(trial_id, set(valid_frames))
    broken_set = {idx for idx, ann in store.annotations.get(trial_id, {}).items()
                  if ann.broken}
    valid_frames = [f for f in valid_frames if f not in broken_set]

    # Get progress
    progress = store.get_trial_progress(trial_id, valid_frames)

    return jsonify({
        'trial_id': trial_id,
        'dataset': trial.dataset,
        'trial_name': trial.trial_name,
        'total_frames': trial.total_frames,
        'sampled_frames': valid_frames,
        'progress': progress
    })


@api_bp.route('/trials/<path:trial_id>/frames_lite', methods=['GET'])
def get_trial_frames_lite(trial_id: str):
    """Get sampled frames for a trial WITHOUT loading all annotations.

    Returns frame indices, lightweight progress summary, and annotation file
    count. This is the fast-start endpoint used during progressive trial
    loading so the first frame can be shown immediately.

    Args:
        trial_id: Trial identifier (e.g., "7DOF2024/Trial1")
    """
    fm = get_frame_manager()
    store = get_annotation_store()

    trial = fm.get_trial_info(trial_id)
    if not trial:
        logger.error(f"Trial not found: '{trial_id}'")
        abort(404, description=f"Trial not found: {trial_id}")

    # Get sampled frames (cached after first call — no annotation loading)
    valid_frames = fm.get_sampled_frames(trial_id)

    # Filter out broken frames (same as /frames endpoint)
    store.load_trial_sampled(trial_id, set(valid_frames))
    broken_set = {idx for idx, ann in store.annotations.get(trial_id, {}).items()
                  if ann.broken}
    valid_frames = [f for f in valid_frames if f not in broken_set]

    # Read tiny _progress.json (no full load)
    progress = store.load_progress_summary(trial_id)

    # Count annotation files on disk for sampled frames only
    annotation_file_count = store.count_annotation_files_sampled(
        trial_id, set(valid_frames)
    )

    return jsonify({
        'trial_id': trial_id,
        'dataset': trial.dataset,
        'trial_name': trial.trial_name,
        'total_frames': trial.total_frames,
        'sampled_frames': valid_frames,
        'progress': progress,
        'annotation_file_count': annotation_file_count
    })


@api_bp.route('/frames/<path:trial_id>/<int:frame_idx>/annotation_single', methods=['GET'])
def get_frame_annotation_single(trial_id: str, frame_idx: int):
    """Get annotation for a single frame without loading the full trial.

    Reads only the one frame JSON file. Does not provide prior annotation
    (that requires the full trial cache). Used for quick first-frame display.

    Args:
        trial_id: Trial identifier
        frame_idx: Frame index
    """
    fm = get_frame_manager()
    store = get_annotation_store()

    # Read single frame annotation (no full trial load)
    annotation = store.get_single_frame(trial_id, frame_idx)

    # Get kinematics data (cached by fm)
    kinematics = fm.load_kinematics(trial_id)
    tool_kinematics = None
    if kinematics:
        annotations = kinematics.get('annotations', [])
        frame_lookup = {ann.get('Frame', i): ann for i, ann in enumerate(annotations)}
        frame_data = frame_lookup.get(frame_idx)
        if frame_data:
            tool_kinematics = {
                'tool1': frame_data.get('Tool 1', {}),
                'tool2': frame_data.get('Tool 2', {}),
                'camera': frame_data.get('Camera', {}),
                'world': frame_data.get('World', {})
            }

    return jsonify({
        'frame_idx': frame_idx,
        'annotation': annotation.to_dict(),
        'prior': None,
        'is_complete': annotation.is_complete(),
        'kinematics': tool_kinematics
    })


@api_bp.route('/trials/load_stream', methods=['GET'])
def stream_trial_load():
    """SSE endpoint that streams annotation loading progress.

    Uses query param for trial_id to avoid URL-encoding issues with
    trial IDs containing slashes (e.g., "7DOF2024/Trial34").

    Query params:
        trial_id: Trial identifier

    SSE events:
        progress: {loaded, total} — sent every 10 files
        done: {loaded, total, cached} — sent once at completion
    """
    trial_id = request.args.get('trial_id')
    if not trial_id:
        abort(400, description="trial_id query parameter required")

    full = request.args.get('full', 'false') == 'true'

    fm = get_frame_manager()
    store = get_annotation_store()
    sampled = set(fm.get_sampled_frames(trial_id))

    def generate():
        was_cached = trial_id in store.annotations
        last_sent = 0
        loaded = 0
        total = 0

        if full:
            # Merge remaining frames into existing cache (no pop — race-safe)
            gen = store.load_remaining_frames_progressive(trial_id)
        else:
            gen = store.load_trial_progressive_sampled(trial_id, sampled)

        for loaded, total in gen:
            # Throttle: send every 10 files or on completion
            if loaded - last_sent >= 10 or loaded == total:
                yield f"event: progress\ndata: {json.dumps({'loaded': loaded, 'total': total})}\n\n"
                last_sent = loaded

        yield f"event: done\ndata: {json.dumps({'loaded': loaded, 'total': total, 'cached': was_cached})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


@api_bp.route('/frames/<path:trial_id>/<int:frame_idx>/image', methods=['GET'])
def get_frame_image(trial_id: str, frame_idx: int):
    """Get frame image file.

    Args:
        trial_id: Trial identifier
        frame_idx: Frame index
    """
    fm = get_frame_manager()
    frame_path = fm.get_frame_path(trial_id, frame_idx)

    if not frame_path or not frame_path.exists():
        abort(404, description=f"Frame not found: {trial_id}/{frame_idx}")

    mimetype = 'image/png' if frame_path.suffix.lower() == '.png' else 'image/bmp'
    response = send_file(frame_path, mimetype=mimetype)
    response.headers['Cache-Control'] = 'public, max-age=2592000, immutable'
    return response


@api_bp.route('/frames/<path:trial_id>/<int:frame_idx>/thumbnail', methods=['GET'])
def get_frame_thumbnail(trial_id: str, frame_idx: int):
    """Get a low-resolution thumbnail for batch annotation view.

    Args:
        trial_id: Trial identifier
        frame_idx: Frame index

    Query params:
        width: Thumbnail width (default 150)
        quality: JPEG quality 0-100 (default 70)

    Returns:
        JPEG thumbnail image
    """
    fm = get_frame_manager()
    frame_path = fm.get_frame_path(trial_id, frame_idx)

    if not frame_path or not frame_path.exists():
        abort(404, description=f"Frame not found: {trial_id}/{frame_idx}")

    # Get optional params
    width = request.args.get('width', 150, type=int)
    quality = request.args.get('quality', 70, type=int)

    try:
        img = Image.open(frame_path)
        # Calculate height to maintain aspect ratio
        aspect = img.height / img.width
        height = int(width * aspect)
        img.thumbnail((width, height), Image.Resampling.LANCZOS)

        # Convert to RGB if necessary (for JPEG)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save to bytes buffer
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)

        response = send_file(buffer, mimetype='image/jpeg')
        # Frames never change — cache aggressively (30 days) so browser
        # serves from disk cache on page refresh / reopen.
        response.headers['Cache-Control'] = 'public, max-age=2592000, immutable'
        return response
    except Exception as e:
        logger.error(f"Failed to create thumbnail for {trial_id}/{frame_idx}: {e}")
        abort(500, description="Failed to create thumbnail")


@api_bp.route('/frames/<path:trial_id>/<int:frame_idx>/annotations', methods=['GET'])
def get_frame_annotations(trial_id: str, frame_idx: int):
    """Get annotations for a frame.

    Args:
        trial_id: Trial identifier
        frame_idx: Frame index
    """
    fm = get_frame_manager()
    store = get_annotation_store()

    # Get current annotation
    annotation = store.get_frame(trial_id, frame_idx)

    # Get prior annotation if available
    valid_frames = fm.get_sampled_frames(trial_id)
    prior = store.get_prior_annotation(trial_id, frame_idx, valid_frames)

    # Get kinematics data for this frame (tool positions and rotations)
    kinematics = fm.load_kinematics(trial_id)
    tool_kinematics = None
    if kinematics:
        annotations = kinematics.get('annotations', [])
        frame_lookup = {ann.get('Frame', i): ann for i, ann in enumerate(annotations)}
        frame_data = frame_lookup.get(frame_idx)
        if frame_data:
            tool_kinematics = {
                'tool1': frame_data.get('Tool 1', {}),
                'tool2': frame_data.get('Tool 2', {}),
                'camera': frame_data.get('Camera', {}),
                'world': frame_data.get('World', {})
            }

    return jsonify({
        'frame_idx': frame_idx,
        'annotation': annotation.to_dict(),
        'prior': prior.to_dict() if prior else None,
        'is_complete': annotation.is_complete(),
        'kinematics': tool_kinematics
    })


@api_bp.route('/frames/<path:trial_id>/<int:frame_idx>/annotations', methods=['POST'])
def save_frame_annotations(trial_id: str, frame_idx: int):
    """Save annotations for a frame.

    Args:
        trial_id: Trial identifier
        frame_idx: Frame index
    """
    store = get_annotation_store()

    data = request.get_json()
    if not data:
        abort(400, description="No data provided")

    # Debug logging: show what keys were received and keypoint data
    logger.info(f"Save request for {trial_id}/{frame_idx}: keys={list(data.keys())}")
    for key in ['tool1_joint', 'tool1_ee_tip', 'tool2_joint', 'tool2_ee_tip']:
        if key in data:
            logger.info(f"  {key}: {data[key]}")

    # Migrate old tool*_missing to tool*_visibility if sent by older clients
    for tool_num in [1, 2]:
        old_key = f'tool{tool_num}_missing'
        new_key = f'tool{tool_num}_visibility'
        if old_key in data and new_key not in data:
            missing = data[old_key]
            data[new_key] = {
                'mask': 0 if missing.get('mask') else 1,
                'lines': 0 if missing.get('lines') else 1
            }
            del data[old_key]

    # Update annotation
    updates = {}
    for key in ['tool1_mask', 'tool2_mask', 'tool1_lines', 'tool2_lines',
                'tool1_joint', 'tool1_ee_tip', 'tool1_ee_left', 'tool1_ee_right',
                'tool2_joint', 'tool2_ee_tip', 'tool2_ee_left', 'tool2_ee_right',
                'tool1_visibility', 'tool2_visibility',
                'skipped', 'broken', 'exclude',
                'pegs', 'pegboard', 'phase']:
        if key in data:
            updates[key] = data[key]

    annotation = store.update_frame(trial_id, frame_idx, updates)
    _invalidate_progress_cache()

    return jsonify({
        'success': True,
        'annotation': annotation.to_dict(),
        'is_complete': annotation.is_complete()
    })


@api_bp.route('/frames/<path:trial_id>/<int:frame_idx>/skip', methods=['POST'])
def skip_frame(trial_id: str, frame_idx: int):
    """Mark a frame as skipped.

    Args:
        trial_id: Trial identifier
        frame_idx: Frame index
    """
    store = get_annotation_store()
    annotation = store.skip_frame(trial_id, frame_idx)
    _invalidate_progress_cache()

    return jsonify({
        'success': True,
        'annotation': annotation.to_dict()
    })


@api_bp.route('/frames/<path:trial_id>/<int:frame_idx>/unskip', methods=['POST'])
def unskip_frame(trial_id: str, frame_idx: int):
    """Remove skipped status from a frame.

    Args:
        trial_id: Trial identifier
        frame_idx: Frame index
    """
    store = get_annotation_store()
    annotation = store.update_frame(trial_id, frame_idx, {'skipped': False})

    return jsonify({
        'success': True,
        'annotation': annotation.to_dict()
    })


@api_bp.route('/frames/<path:trial_id>/<int:frame_idx>/ensure_init', methods=['POST'])
def ensure_frame_init(trial_id: str, frame_idx: int):
    """Ensure annotation file exists on disk, creating a blank one if needed."""
    store = get_annotation_store()

    frame_path = store._get_frame_path(trial_id, frame_idx)
    initialized = False

    if not frame_path.exists():
        store.get_frame(trial_id, frame_idx)
        store.save_frame(trial_id, frame_idx)
        initialized = True

    return jsonify({'initialized': initialized, 'frame_idx': frame_idx})


@api_bp.route('/trials/<path:trial_id>/progress', methods=['GET'])
def get_trial_progress(trial_id: str):
    """Get annotation progress for a trial.

    Args:
        trial_id: Trial identifier
    """
    fm = get_frame_manager()
    store = get_annotation_store()

    valid_frames = fm.get_sampled_frames(trial_id)
    progress = store.get_trial_progress(trial_id, valid_frames)

    return jsonify(progress)


@api_bp.route('/trials/<path:trial_id>/frame_status', methods=['GET'])
def get_frame_status(trial_id: str):
    """Get per-frame completion status for all backend-sampled frames.

    Args:
        trial_id: Trial identifier (e.g., "7DOF2024/Trial1")

    Returns:
        JSON dict mapping frame index (string) to status string:
        'completed', 'skipped', 'partial', or 'none'.
    """
    fm = get_frame_manager()
    store = get_annotation_store()

    valid_frames = fm.get_sampled_frames(trial_id)
    store.load_trial_sampled(trial_id, set(valid_frames))
    frames = store.annotations.get(trial_id, {})

    valid_set = set(valid_frames)
    status = {}
    for frame_idx in valid_frames:
        ann = frames.get(frame_idx)
        if not ann:
            status[str(frame_idx)] = 'none'
        elif ann.broken:
            status[str(frame_idx)] = 'broken'
        elif ann.skipped:
            status[str(frame_idx)] = 'skipped'
        elif ann.is_negative():
            status[str(frame_idx)] = 'negative'
        elif ann.is_complete():
            status[str(frame_idx)] = 'completed'
        else:
            status[str(frame_idx)] = 'partial'

    # Include completed/skipped/broken/negative off-sample frames
    for frame_idx, ann in frames.items():
        if frame_idx not in valid_set:
            if ann.broken:
                status[str(frame_idx)] = 'broken'
            elif ann.skipped:
                status[str(frame_idx)] = 'skipped'
            elif ann.is_negative():
                status[str(frame_idx)] = 'negative'
            elif ann.is_complete():
                status[str(frame_idx)] = 'completed'

    return jsonify(status)


@api_bp.route('/navigation/<path:trial_id>/<int:current_frame>', methods=['GET'])
def get_navigation(trial_id: str, current_frame: int):
    """Get navigation info for current frame.

    Args:
        trial_id: Trial identifier
        current_frame: Current frame index
    """
    fm = get_frame_manager()

    next_frame = fm.get_next_frame(trial_id, current_frame)
    prev_frame = fm.get_prev_frame(trial_id, current_frame)

    trial = fm.get_trial_info(trial_id)
    valid_frames = trial.valid_frames if trial else []

    # Find current index in valid frames
    current_idx = -1
    if current_frame in valid_frames:
        current_idx = valid_frames.index(current_frame)

    return jsonify({
        'current_frame': current_frame,
        'current_idx': current_idx,
        'total_sampled': len(valid_frames),
        'next_frame': next_frame,
        'prev_frame': prev_frame,
        'has_next': next_frame is not None,
        'has_prev': prev_frame is not None
    })


@api_bp.route('/trials/<path:trial_id>/all_frames', methods=['GET'])
def get_all_frames(trial_id: str):
    """Get ALL frame indices for a trial (not just sampled).

    Enumerates actual files on disk and excludes frames marked as broken.

    Args:
        trial_id: Trial identifier
    """
    fm = get_frame_manager()
    store = get_annotation_store()

    trial = fm.get_trial_info(trial_id)
    if not trial:
        abort(404, description=f"Trial not found: {trial_id}")

    # Get actual file-based indices (excludes files in broken/ subfolder)
    all_indices = fm.get_all_frame_indices(trial_id)

    # Also filter out frames marked broken in annotation store
    store.load_trial(trial_id)
    frames = store.annotations.get(trial_id, {})
    all_frames = [idx for idx in all_indices
                  if not (idx in frames and frames[idx].broken)]

    return jsonify({
        'trial_id': trial_id,
        'total_frames': len(all_frames),
        'all_frames': all_frames
    })


@api_bp.route('/datasets/progress', methods=['GET'])
def get_all_progress():
    """Get progress for all datasets and trials with percentages.

    Reads lightweight per-trial ``_progress.json`` summary files instead of
    loading all individual frame JSONs.  This reduces the I/O from ~16K file
    reads to ~100 tiny files, cutting load time from minutes to < 1 second.

    Startup warmup (app.py before_request) seeds summaries and pre-loads
    them into memory, so this endpoint reads from cache instantly.
    """
    fm = get_frame_manager()
    t0 = time.time()
    store = get_annotation_store()

    datasets = fm.discover_all_trials()
    result = []

    for dataset_name, trials in datasets.items():
        dataset_total = 0
        dataset_completed = 0
        trial_progress = []

        for trial_name in trials:
            trial_id = f"{dataset_name}/{trial_name}"
            trial_info = fm.get_trial_info(trial_id)

            # Estimate sampled frame count from total_frames
            estimated_total = (
                (trial_info.total_frames // fm.SAMPLE_INTERVAL) + 1
                if trial_info and trial_info.total_frames > 0 else 0
            )

            # Fast path: read summary instead of loading all frame JSONs
            summary = store.load_progress_summary(trial_id)
            if summary:
                completed = summary.get('completed', 0)
                skipped = summary.get('skipped', 0)
                negative = summary.get('negative', 0)
                partial = summary.get('partial', 0)
                broken = summary.get('broken', 0)
                peg_completed = summary.get('peg_completed', 0)
                phase_completed = summary.get('phase_completed', 0)
                broken_total = summary.get('broken_total', 0)
                excluded_total = summary.get('excluded_total', 0)
            else:
                completed = skipped = negative = partial = broken = 0
                peg_completed = phase_completed = 0
                broken_total = excluded_total = 0

            # Use estimated sampled count as denominator — progress summaries
            # now only count sampled frames so this is always correct
            trial_total = estimated_total
            trial_completed = completed + skipped
            trial_pct = (trial_completed / trial_total * 100) if trial_total > 0 else 0

            # Peg progress shares the sampled grid with tool progress.
            peg_total = trial_total
            peg_pct = (peg_completed / peg_total * 100) if peg_total > 0 else 0

            # Phase progress is over every recorded frame, minus broken/excluded.
            full_total = trial_info.total_frames if trial_info else 0
            phase_total = max(full_total - broken_total - excluded_total, 0)
            phase_pct = (phase_completed / phase_total * 100) if phase_total > 0 else 0

            dataset_total += trial_total
            dataset_completed += trial_completed

            trial_progress.append({
                'trial_id': trial_id,
                'trial_name': trial_name,
                'total': trial_total,
                'completed': completed,
                'skipped': skipped,
                'negative': negative,
                'remaining': trial_total - trial_completed,
                'percentage': round(trial_pct, 1),
                'peg_completed': peg_completed,
                'peg_total': peg_total,
                'peg_percentage': round(peg_pct, 1),
                'phase_completed': phase_completed,
                'phase_total': phase_total,
                'phase_percentage': round(phase_pct, 1),
            })

        dataset_pct = (dataset_completed / dataset_total * 100) if dataset_total > 0 else 0

        result.append({
            'name': dataset_name,
            'total': dataset_total,
            'completed': dataset_completed,
            'percentage': round(dataset_pct, 1),
            'trials': trial_progress
        })

    logger.info(f"Progress scan took {time.time() - t0:.2f}s")
    return jsonify(result)


@api_bp.route('/datasets/<path:dataset_name>/refresh_progress', methods=['GET'])
def refresh_dataset_progress(dataset_name: str):
    """Recompute accurate progress for all trials in a dataset from disk.

    Reads annotation files directly (bypasses in-memory cache), classifies
    each, and writes fresh ``_progress.json`` summaries.  Returns progress
    in the same shape as a single entry from ``/datasets/progress``.

    Args:
        dataset_name: Dataset name (e.g. "7DOF2024").
    """
    fm = get_frame_manager()
    store = get_annotation_store()

    datasets = fm.discover_all_trials()
    trials = datasets.get(dataset_name)
    if trials is None:
        abort(404, description=f"Dataset not found: {dataset_name}")

    t0 = time.time()
    dataset_total = 0
    dataset_completed = 0
    trial_progress = []

    for trial_name in trials:
        trial_id = f"{dataset_name}/{trial_name}"
        sampled = set(fm.get_sampled_frames(trial_id))
        trial_total = len(sampled)

        counts = store.refresh_trial_progress_from_files(trial_id, sampled)
        completed = counts.get('completed', 0)
        skipped = counts.get('skipped', 0)
        negative = counts.get('negative', 0)
        peg_completed = counts.get('peg_completed', 0)
        phase_completed = counts.get('phase_completed', 0)
        broken_total = counts.get('broken_total', 0)
        excluded_total = counts.get('excluded_total', 0)
        trial_done = completed + skipped
        trial_pct = (trial_done / trial_total * 100) if trial_total > 0 else 0

        peg_total = trial_total
        peg_pct = (peg_completed / peg_total * 100) if peg_total > 0 else 0

        trial_info = fm.get_trial_info(trial_id)
        full_total = trial_info.total_frames if trial_info else 0
        phase_total = max(full_total - broken_total - excluded_total, 0)
        phase_pct = (phase_completed / phase_total * 100) if phase_total > 0 else 0

        dataset_total += trial_total
        dataset_completed += trial_done

        trial_progress.append({
            'trial_id': trial_id,
            'trial_name': trial_name,
            'total': trial_total,
            'completed': completed,
            'skipped': skipped,
            'negative': negative,
            'remaining': trial_total - trial_done,
            'percentage': round(trial_pct, 1),
            'peg_completed': peg_completed,
            'peg_total': peg_total,
            'peg_percentage': round(peg_pct, 1),
            'phase_completed': phase_completed,
            'phase_total': phase_total,
            'phase_percentage': round(phase_pct, 1),
        })

    dataset_pct = (dataset_completed / dataset_total * 100) if dataset_total > 0 else 0

    logger.info(
        f"Refreshed progress for {dataset_name} ({len(trials)} trials) "
        f"in {time.time() - t0:.2f}s"
    )

    return jsonify({
        'name': dataset_name,
        'total': dataset_total,
        'completed': dataset_completed,
        'percentage': round(dataset_pct, 1),
        'trials': trial_progress,
    })


@api_bp.route('/sam/precomputed/<path:trial_id>/frame/<int:frame_idx>', methods=['GET'])
def get_precomputed_sam(trial_id: str, frame_idx: int):
    """Get pre-computed SAM segments for a frame.

    Args:
        trial_id: Trial identifier (e.g., "7DOF2024/Attempt 1 .../Trial1")
        frame_idx: Frame index

    Returns:
        JSON with polygons, scores, areas, bboxes and available flag.
    """
    fm = get_frame_manager()
    npz_path = fm.get_sam_path(trial_id, frame_idx)

    if not npz_path:
        return jsonify({'available': False})

    try:
        data = np.load(npz_path, allow_pickle=True)

        polygons_raw = data['polygons']
        scores = data['scores'].tolist()
        areas = data['areas'].tolist()
        bboxes = data['bboxes'].tolist()

        # Convert object array of polygons to list of lists
        polygons = []
        for poly in polygons_raw:
            if poly is not None and len(poly) >= 3:
                polygons.append([[float(pt[0]), float(pt[1])] for pt in poly])

        return jsonify({
            'available': True,
            'polygons': polygons,
            'scores': scores[:len(polygons)],
            'areas': areas[:len(polygons)],
            'bboxes': bboxes[:len(polygons)],
        })

    except Exception as e:
        logger.error(f"Failed to load precomputed SAM for {trial_id}/{frame_idx}: {e}")
        return jsonify({'available': False})


@api_bp.route('/sam/compute/<path:trial_id>/frame/<int:frame_idx>', methods=['POST'])
def compute_sam_on_demand(trial_id: str, frame_idx: int):
    """Compute SAM automatic masks on-demand for a single frame.

    If precomputed .npz already exists, returns it directly. Otherwise
    runs automatic mask generation, saves the .npz, and returns the result.

    Args:
        trial_id: Trial identifier (e.g., "7DOF2024/Attempt 1 .../Trial1")
        frame_idx: Frame index

    Returns:
        JSON with polygons, scores, areas, bboxes and available flag.
    """
    fm = get_frame_manager()

    # Check if already precomputed
    npz_path = fm.get_sam_path(trial_id, frame_idx)
    if npz_path:
        # Already exists — return it (same logic as get_precomputed_sam)
        try:
            data = np.load(npz_path, allow_pickle=True)
            polygons_raw = data['polygons']
            scores = data['scores'].tolist()
            areas = data['areas'].tolist()
            bboxes = data['bboxes'].tolist()

            polygons = []
            for poly in polygons_raw:
                if poly is not None and len(poly) >= 3:
                    polygons.append([[float(pt[0]), float(pt[1])] for pt in poly])

            return jsonify({
                'available': True,
                'polygons': polygons,
                'scores': scores[:len(polygons)],
                'areas': areas[:len(polygons)],
                'bboxes': bboxes[:len(polygons)],
            })
        except Exception as e:
            logger.error(f"Failed to load existing SAM for {trial_id}/{frame_idx}: {e}")
            # Fall through to recompute

    # Load the frame image
    frame_path = fm.get_frame_path(trial_id, frame_idx)
    if not frame_path or not frame_path.exists():
        abort(404, description=f"Frame not found: {trial_id}/{frame_idx}")

    try:
        image = np.array(Image.open(frame_path).convert('RGB'))
    except Exception as e:
        logger.error(f"Failed to load image {frame_path}: {e}")
        abort(500, description="Failed to load frame image")

    # Run automatic mask generation
    masks_data = sam_segmentation.generate_automatic_masks(image)
    if not masks_data:
        return jsonify({'available': False})

    # Sort by area descending (largest first) — same as precompute_sam.py
    masks_data.sort(key=lambda m: m["area"], reverse=True)

    # Convert masks to polygons and save .npz
    polygons = []
    scores = []
    areas = []
    bboxes = []

    for mask_info in masks_data:
        poly = sam_segmentation._mask_to_polygon(mask_info["segmentation"])
        if len(poly) < 3:
            continue
        polygons.append(poly)
        scores.append(float(mask_info.get("predicted_iou", mask_info.get("stability_score", 0.0))))
        areas.append(int(mask_info["area"]))
        bbox = mask_info["bbox"]  # [x, y, w, h] from SAM
        bboxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

    # Save .npz for future use
    trial_info = fm.get_trial_info(trial_id)
    if trial_info:
        sam_dir = trial_info.trial_path / "SAM"
        sam_dir.mkdir(exist_ok=True)
        save_path = sam_dir / f"frame_{frame_idx:05d}.npz"

        poly_array = np.empty(len(polygons), dtype=object)
        for j, p in enumerate(polygons):
            poly_array[j] = p

        np.savez_compressed(
            save_path,
            polygons=poly_array,
            scores=np.array(scores, dtype=np.float32),
            areas=np.array(areas, dtype=np.int32),
            bboxes=np.array(bboxes, dtype=np.float32) if bboxes else np.array([], dtype=np.float32).reshape(0, 4),
        )
        logger.info(f"Saved on-demand SAM masks for {trial_id}/frame_{frame_idx:05d}")

    # Return result
    polygons_json = [[[float(pt[0]), float(pt[1])] for pt in poly] for poly in polygons]

    return jsonify({
        'available': True,
        'polygons': polygons_json,
        'scores': scores,
        'areas': areas,
        'bboxes': bboxes,
    })


@api_bp.route('/sam/availability/<path:trial_id>', methods=['GET'])
def get_sam_availability(trial_id: str):
    """Check which frames have precomputed SAM masks for a trial.

    Args:
        trial_id: Trial identifier

    Query params:
        frames: Comma-separated frame indices to check. If omitted,
                checks all sampled frames for the trial.

    Returns:
        JSON with dict mapping frame index (string) to boolean.
    """
    fm = get_frame_manager()

    frames_param = request.args.get('frames')
    if frames_param:
        frame_indices = [int(f) for f in frames_param.split(',') if f.strip()]
    else:
        frame_indices = fm.get_sampled_frames(trial_id)

    trial_info = fm.get_trial_info(trial_id)
    if not trial_info:
        abort(404, description=f"Trial not found: {trial_id}")

    sam_dir = trial_info.trial_path / "SAM"
    availability = {}
    for idx in frame_indices:
        npz_path = sam_dir / f"frame_{idx:05d}.npz"
        availability[str(idx)] = npz_path.exists()

    return jsonify(availability)


@api_bp.route('/sam/status', methods=['GET'])
def sam_status():
    """Get SAM model status (available, loaded, device)."""
    return jsonify(sam_segmentation.get_status())


@api_bp.route('/sam/segment/<path:trial_id>/<int:frame_idx>', methods=['POST'])
def sam_segment(trial_id: str, frame_idx: int):
    """Run SAM segmentation on a frame with point prompts.

    Args:
        trial_id: Trial identifier
        frame_idx: Frame index

    Request body (JSON):
        point_coords: List of [x, y] pixel coordinates
        point_labels: List of labels (1=foreground, 0=background)
        multimask: Whether to return multiple mask proposals (default True)

    Returns:
        JSON with masks (polygon vertices), scores, and best_idx.
    """
    data = request.get_json()
    if not data or 'point_coords' not in data:
        abort(400, description="point_coords required")

    point_coords = data['point_coords']
    point_labels = data.get('point_labels', [1] * len(point_coords))
    multimask = data.get('multimask', True)

    # Load the frame image
    fm = get_frame_manager()
    frame_path = fm.get_frame_path(trial_id, frame_idx)

    if not frame_path or not frame_path.exists():
        abort(404, description=f"Frame not found: {trial_id}/{frame_idx}")

    try:
        image = np.array(Image.open(frame_path).convert('RGB'))
    except Exception as e:
        logger.error(f"Failed to load image {frame_path}: {e}")
        abort(500, description="Failed to load frame image")

    # Set the image (cached if same frame)
    if not sam_segmentation.set_image(image):
        abort(503, description="SAM model not available. Install with: pip install sam2")

    # Run segmentation
    result = sam_segmentation.segment_point(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask=multimask,
    )

    if result is None:
        abort(500, description="SAM segmentation failed")

    return jsonify(result)


@api_bp.route('/frames/<path:trial_id>/visibility', methods=['GET'])
def get_all_frame_visibility(trial_id: str):
    """Get visibility status for all frames in a trial.

    Args:
        trial_id: Trial identifier

    Returns:
        JSON dict mapping frame index (string) to visibility info:
        { "frameIdx": { "t1": {"mask":1|0|-1, ...}, "t2": {...} } }
    """
    store = get_annotation_store()
    store.load_trial(trial_id)
    frames = store.annotations.get(trial_id, {})

    visibility = {}
    for frame_idx, ann in frames.items():
        entry = {
            't1': ann.tool1_visibility,
            't2': ann.tool2_visibility
        }
        if ann.exclude:
            entry['exclude'] = True
        visibility[str(frame_idx)] = entry

    return jsonify(visibility)


@api_bp.route('/frames/<path:trial_id>/batch', methods=['POST'])
def batch_update_frames(trial_id: str):
    """Update multiple frames at once (batch annotation).

    Args:
        trial_id: Trial identifier

    Request body (JSON):
        frame_indices: List of frame indices to update
        updates: Dict of fields to update (e.g., {tool1_visibility: {...}})

    Returns:
        JSON with success status and updated frame count
    """
    store = get_annotation_store()

    data = request.get_json()
    if not data:
        abort(400, description="No data provided")

    frame_indices = data.get('frame_indices', [])
    updates = data.get('updates', {})

    if not frame_indices:
        abort(400, description="No frame_indices provided")

    if not updates:
        abort(400, description="No updates provided")

    # Update each frame without auto-save
    updated_count = 0
    for frame_idx in frame_indices:
        try:
            store.update_frame(trial_id, frame_idx, updates, auto_save=False)
            updated_count += 1
        except Exception as e:
            logger.error(f"Failed to update frame {frame_idx}: {e}")

    # Single atomic save
    saved = store.save_trial(trial_id)
    _invalidate_progress_cache()

    return jsonify({
        'success': saved,
        'updated_count': updated_count,
        'total_requested': len(frame_indices)
    })


@api_bp.route('/trials/<path:trial_id>/backup', methods=['POST'])
def backup_trial(trial_id: str):
    """Create a datetime-stamped backup of all annotations for a trial.

    Args:
        trial_id: Trial identifier
    """
    store = get_annotation_store()
    backup_path = store.backup_trial(trial_id)

    if backup_path is None:
        abort(400, description=f"No annotations to backup for {trial_id}")

    return jsonify({
        'success': True,
        'path': str(backup_path.relative_to(backup_path.parent.parent))
    })


@api_bp.route('/frames/<path:trial_id>/batch-broken', methods=['POST'])
def batch_mark_broken(trial_id: str):
    """Move selected frames to broken/ folder and mark them as broken.

    Args:
        trial_id: Trial identifier

    Request body (JSON):
        frame_indices: List of frame indices to mark as broken

    Returns:
        JSON with success status, moved count, and list of failed indices
    """
    fm = get_frame_manager()
    store = get_annotation_store()

    data = request.get_json()
    if not data:
        abort(400, description="No data provided")

    frame_indices = data.get('frame_indices', [])
    if not frame_indices:
        abort(400, description="No frame_indices provided")

    moved_count = 0
    failed = []

    for frame_idx in frame_indices:
        if fm.move_to_broken(trial_id, frame_idx):
            store.update_frame(trial_id, frame_idx, {'broken': True}, auto_save=False)
            moved_count += 1
        else:
            failed.append(frame_idx)

    # Single atomic save
    store.save_trial(trial_id)
    _invalidate_progress_cache()

    return jsonify({
        'success': moved_count > 0,
        'moved_count': moved_count,
        'failed': failed
    })


# ============================================================================
# Phase & Object Annotation Endpoints
# ============================================================================

@api_bp.route('/phase_definitions', methods=['GET'])
def phase_definitions():
    """Return the full annotation taxonomy for the frontend."""
    return jsonify(get_definitions_dict())


@api_bp.route('/trials/<path:trial_id>/phase_summary', methods=['GET'])
def get_phase_summary(trial_id: str):
    """Return per-frame coarse phase labels for a trial.

    Reads directly from per-frame JSON files on disk so the summary covers
    every annotated frame, not just the sampled ones loaded in memory.

    Returns:
        JSON dict mapping frame_idx (str) -> {coarse, cycle_index}.
        Only includes frames that have a non-empty coarse phase.
    """
    store = get_annotation_store()
    summary: dict[str, dict] = {}

    trial_dir = store._get_trial_dir(trial_id)
    if trial_dir.is_dir():
        for path in trial_dir.glob('frame_*.json'):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue
            phase = data.get('phase') or {}
            if not isinstance(phase, dict):
                continue
            tool1 = phase.get('tool1')
            tool2 = phase.get('tool2')
            coarse = phase.get('coarse', '') or ''
            # Legacy migration on the fly
            if tool1 is None and tool2 is None:
                if not coarse:
                    continue
                at = phase.get('active_tool', 0)
                tool1 = coarse if at != 2 else 'idle'
                tool2 = coarse if at == 2 else 'idle'
            tool1 = tool1 or 'idle'
            tool2 = tool2 or 'idle'
            # Include frame if either tool is non-idle, OR it was explicitly
            # annotated (legacy coarse non-empty, includes explicit 'idle').
            if tool1 == 'idle' and tool2 == 'idle' and not coarse:
                continue
            try:
                frame_idx = int(path.stem.split('_')[1])
            except (IndexError, ValueError):
                continue
            summary[str(frame_idx)] = {
                'tool1': tool1,
                'tool2': tool2,
                'coarse': tool1 if tool1 != 'idle' else tool2,
                'cycle_index': phase.get('cycle_index', 0),
            }

    # Fallback to in-memory (monolithic legacy trials)
    if not summary:
        trial_data = store.annotations.get(trial_id, {})
        for frame_idx, ann in trial_data.items():
            phase = ann.phase if hasattr(ann, 'phase') else {}
            if isinstance(phase, dict):
                tool1 = phase.get('tool1') or 'idle'
                tool2 = phase.get('tool2') or 'idle'
                co = phase.get('coarse', '') or ''
                if tool1 == 'idle' and tool2 == 'idle' and co:
                    at = phase.get('active_tool', 0)
                    tool1 = co if at != 2 else 'idle'
                    tool2 = co if at == 2 else 'idle'
                if tool1 == 'idle' and tool2 == 'idle' and not co:
                    continue
                summary[str(frame_idx)] = {
                    'tool1': tool1,
                    'tool2': tool2,
                    'coarse': tool1 if tool1 != 'idle' else tool2,
                    'cycle_index': phase.get('cycle_index', 0),
                }

    return jsonify(summary)


@api_bp.route('/trials/<path:trial_id>/phase_bulk', methods=['POST'])
def set_phase_bulk(trial_id: str):
    """Bulk-set phase labels for frames.

    Request JSON formats:
        {"frames": [int, ...], "phase_data": {...}}
        {"start_frame": int, "end_frame": int, "phase_data": {...}}
        {"operations": [{"start_frame": int, "end_frame": int, "phase_data": {...}}, ...]}
    """
    store = get_annotation_store()
    data = request.get_json()
    if not data:
        abort(400, description="No data provided")

    modified_frames = []

    # Multi-operation batch mode (single request for all clips)
    operations = data.get('operations')
    if operations:
        for op in operations:
            phase_data = op.get('phase_data', {})
            frames = op.get('frames')
            if frames is None:
                start = op.get('start_frame', 0)
                end = op.get('end_frame', 0)
                if start > end:
                    continue
                frames = range(start, end + 1)
            for frame_idx in frames:
                store.update_frame(trial_id, int(frame_idx), {'phase': phase_data}, auto_save=False)
                modified_frames.append(int(frame_idx))
    else:
        # Single operation (backwards compatible)
        phase_data = data.get('phase_data', {})
        frames = data.get('frames')
        if frames is None:
            start = data.get('start_frame', 0)
            end = data.get('end_frame', 0)
            if start > end:
                abort(400, description="start_frame must be <= end_frame")
            frames = range(start, end + 1)
        for frame_idx in frames:
            store.update_frame(trial_id, int(frame_idx), {'phase': phase_data}, auto_save=False)
            modified_frames.append(int(frame_idx))

    # Save only modified frames (not the entire trial)
    for frame_idx in modified_frames:
        store._save_frame_file(trial_id, frame_idx)
    store._save_progress_summary(trial_id)
    _invalidate_progress_cache()

    return jsonify({
        'success': True,
        'updated_count': len(modified_frames),
    })


@api_bp.route('/frames/<path:trial_id>/batch-copy-pegs', methods=['POST'])
def batch_copy_pegs(trial_id: str):
    """Copy peg, pegboard, and post mask data from one frame to many.

    Request JSON:
        {"source_frame": int, "target_frames": [int, ...]}
    """
    store = get_annotation_store()
    data = request.get_json()
    if not data:
        abort(400, description="No data provided")

    source_idx = data.get('source_frame')
    target_indices = data.get('target_frames', [])
    if source_idx is None or not target_indices:
        abort(400, description="source_frame and target_frames required")

    source_ann = store.get_annotation(trial_id, int(source_idx))
    source_pegs = source_ann.pegs
    source_pegboard = source_ann.pegboard

    updated = 0
    for idx in target_indices:
        store.update_frame(trial_id, int(idx), {
            'pegs': copy.deepcopy(source_pegs),
            'pegboard': copy.deepcopy(source_pegboard),
        }, auto_save=False)
        updated += 1

    store.save_trial(trial_id)
    _invalidate_progress_cache()

    return jsonify({
        'success': True,
        'count': updated,
    })


@api_bp.route('/frames/<path:trial_id>/latest-peg-frame', methods=['GET'])
def get_latest_peg_frame(trial_id: str):
    """Find most recent frame before `before` that has peg or pegboard data.

    Query params:
        before (int): frame index to search backwards from
        data_type (str): 'pegs' or 'pegboard' (default: 'pegs')
    """
    store = get_annotation_store()
    fm = get_frame_manager()
    before_idx = request.args.get('before', type=int)
    data_type = request.args.get('data_type', 'pegs')

    if before_idx is None:
        abort(400, description="'before' query parameter required")

    valid_frames = fm.get_sampled_frames(trial_id)
    for frame_idx in reversed(sorted(valid_frames)):
        if frame_idx >= before_idx:
            continue
        ann = store.get_frame(trial_id, frame_idx)

        has_pegs = bool(ann.pegs and len(ann.pegs) > 0)

        pb = ann.pegboard
        has_pegboard = False
        if pb:
            has_posts = any(pt for pt in (pb.get('source_posts') or []) if pt)
            has_masks = any(m for m in (pb.get('source_post_masks') or []) if m and len(m) >= 3)
            has_masks = has_masks or any(m for m in (pb.get('target_post_masks') or []) if m and len(m) >= 3)
            has_board = bool(pb.get('board_mask'))
            has_kps = any(k for k in (pb.get('source_post_keypoints') or []) if k)
            has_kps = has_kps or any(k for k in (pb.get('target_post_keypoints') or []) if k)
            has_pegboard = has_posts or has_masks or has_board or has_kps

        if data_type == 'any':
            if has_pegs or has_pegboard:
                return jsonify({'frame_idx': frame_idx, 'annotation': ann.to_dict()})
        elif data_type == 'pegboard':
            if has_pegboard:
                return jsonify({'frame_idx': frame_idx, 'annotation': ann.to_dict()})
        else:
            if has_pegs:
                return jsonify({'frame_idx': frame_idx, 'annotation': ann.to_dict()})

    return jsonify({'frame_idx': None, 'annotation': None})
