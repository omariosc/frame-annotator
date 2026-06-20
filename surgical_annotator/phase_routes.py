"""Flask blueprint for phase annotation/validation tool.

Serves API endpoints for reviewing algorithm-predicted phase boundaries
against kinematic signals (speed, jaw angle, inter-tool distance).

Templates live in: outputs/papers/paper3/results/phase_labels/templates/
"""

import json
import re
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request, send_from_directory

phase_bp = Blueprint('phase', __name__)

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = PROJECT_ROOT / 'outputs' / 'papers' / 'paper3' / 'results' / 'phase_labels' / 'templates'

# ── Constants ───────────────────────────────────────────────────────────────
JAW_MAX_RAD = 0.6981  # ~40 degrees max jaw opening


def _load_json_with_trailing_commas(path: Path) -> dict:
    """Load JSON file, fixing trailing commas (common in our label.json files)."""
    text = path.read_text(encoding='utf-8')
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return json.loads(text)


def _calibrate_voltage_to_radians(
    raw_voltage: np.ndarray,
    lower_pct: float = 10.0,
    upper_pct: float = 90.0,
    max_jaw_rad: float = JAW_MAX_RAD,
) -> np.ndarray:
    """Convert raw jaw voltage to radians using percentile mapping."""
    if len(raw_voltage) == 0:
        return np.array([])
    v_low = np.percentile(raw_voltage, lower_pct)
    v_high = np.percentile(raw_voltage, upper_pct)
    if v_high <= v_low:
        return np.zeros_like(raw_voltage)
    calibrated = (raw_voltage - v_low) / (v_high - v_low) * max_jaw_rad
    return np.clip(calibrated, 0.0, max_jaw_rad)


def _compute_signals(label_data: dict, fps: float) -> dict:
    """Compute kinematic signals from label.json annotations.

    Returns dict with arrays for: tool1_speed, tool2_speed,
    tool1_jaw, tool2_jaw, inter_tool_dist, time_s.
    """
    annotations = label_data['annotations']
    n = len(annotations)

    # Extract positions and angles
    t1_pos = np.zeros((n, 3))
    t2_pos = np.zeros((n, 3))
    t1_angle = np.zeros(n)
    t2_angle = np.zeros(n)
    time_s = np.zeros(n)

    for i, ann in enumerate(annotations):
        time_s[i] = ann['Time']
        t1 = ann.get('Tool 1', {})
        t2 = ann.get('Tool 2', {})
        if 'Position' in t1:
            t1_pos[i] = t1['Position']
        if 'Position' in t2:
            t2_pos[i] = t2['Position']
        if 'Angle' in t1:
            t1_angle[i] = t1['Angle']
        if 'Angle' in t2:
            t2_angle[i] = t2['Angle']

    # Speed: norm of velocity (mm/s)
    dt = np.diff(time_s)
    dt[dt <= 0] = 1.0 / fps  # avoid division by zero

    t1_vel = np.diff(t1_pos, axis=0)
    t2_vel = np.diff(t2_pos, axis=0)
    t1_speed = np.linalg.norm(t1_vel, axis=1) / dt
    t2_speed = np.linalg.norm(t2_vel, axis=1) / dt
    # Pad first frame with 0
    t1_speed = np.concatenate([[0], t1_speed])
    t2_speed = np.concatenate([[0], t2_speed])

    # Jaw angle (calibrated to radians)
    t1_jaw = _calibrate_voltage_to_radians(t1_angle)
    t2_jaw = _calibrate_voltage_to_radians(t2_angle)

    # Inter-tool distance
    inter_dist = np.linalg.norm(t1_pos - t2_pos, axis=1)

    # Downsample for frontend if > 5000 frames (keep responsive)
    if n > 5000:
        step = n // 5000
        indices = np.arange(0, n, step)
        return {
            'tool1_speed': t1_speed[indices].tolist(),
            'tool2_speed': t2_speed[indices].tolist(),
            'tool1_jaw': t1_jaw[indices].tolist(),
            'tool2_jaw': t2_jaw[indices].tolist(),
            'inter_tool_dist': inter_dist[indices].tolist(),
            'time_s': time_s[indices].tolist(),
            'frame_indices': indices.tolist(),
            'downsampled': True,
            'step': int(step),
        }

    return {
        'tool1_speed': t1_speed.tolist(),
        'tool2_speed': t2_speed.tolist(),
        'tool1_jaw': t1_jaw.tolist(),
        'tool2_jaw': t2_jaw.tolist(),
        'inter_tool_dist': inter_dist.tolist(),
        'time_s': time_s.tolist(),
        'frame_indices': list(range(n)),
        'downsampled': False,
        'step': 1,
    }


def _find_label_json(trial_key: str) -> Path | None:
    """Find the label.json file for a given trial key.

    trial_key format: '7DOF2024/Attempt 1 - Day 1 with Latency/Trial1'
    """
    label_path = PROJECT_ROOT / trial_key / 'label.json'
    if label_path.exists():
        return label_path
    return None


def _list_templates() -> list[dict]:
    """List all template files with completion status."""
    if not TEMPLATE_DIR.exists():
        return []
    templates = []
    for f in sorted(TEMPLATE_DIR.glob('template_*.json')):
        data = json.loads(f.read_text(encoding='utf-8'))
        preds = data.get('algorithm_predictions', [])
        n_total = len(preds)
        n_done = sum(1 for p in preds if p.get('correct') is not None)
        templates.append({
            'filename': f.name,
            'trial_key': data.get('trial_key', ''),
            'dataset': data.get('dataset', ''),
            'skill_category': data.get('skill_category', ''),
            'n_frames': data.get('n_frames', 0),
            'fps': data.get('fps', 0),
            'total_time_s': data.get('total_time_s', 0),
            'n_predictions': n_total,
            'n_validated': n_done,
            'complete': n_done == n_total and n_total > 0,
        })
    return templates


# ── Routes ──────────────────────────────────────────────────────────────────

@phase_bp.route('/phase')
def phase_index():
    """Serve the phase annotation page."""
    static_dir = Path(__file__).parent / 'static'
    return send_from_directory(str(static_dir), 'phase.html')


@phase_bp.route('/api/phase/templates')
def list_templates():
    """GET: List all available phase annotation templates."""
    return jsonify(_list_templates())


@phase_bp.route('/api/phase/template/<path:filename>')
def get_template(filename: str):
    """GET: Load a template with computed kinematic signals."""
    template_path = TEMPLATE_DIR / filename
    if not template_path.exists():
        return jsonify({'error': f'Template not found: {filename}'}), 404

    template_data = json.loads(template_path.read_text(encoding='utf-8'))
    trial_key = template_data.get('trial_key', '')
    fps = template_data.get('fps', 13.0)

    # Load kinematic signals from label.json
    label_path = _find_label_json(trial_key)
    signals = None
    if label_path:
        label_data = _load_json_with_trailing_commas(label_path)
        signals = _compute_signals(label_data, fps)

    return jsonify({
        'template': template_data,
        'signals': signals,
    })


@phase_bp.route('/api/phase/template/<path:filename>', methods=['POST'])
def save_template(filename: str):
    """POST: Save corrections back to template file."""
    template_path = TEMPLATE_DIR / filename
    if not template_path.exists():
        return jsonify({'error': f'Template not found: {filename}'}), 404

    updates = request.get_json()
    if not updates:
        return jsonify({'error': 'No JSON body provided'}), 400

    # Load current template
    template_data = json.loads(template_path.read_text(encoding='utf-8'))

    # Update algorithm_predictions if provided
    if 'algorithm_predictions' in updates:
        template_data['algorithm_predictions'] = updates['algorithm_predictions']

    # Update manual_corrections if provided
    if 'manual_corrections' in updates:
        template_data['manual_corrections'] = updates['manual_corrections']

    # Write back
    template_path.write_text(
        json.dumps(template_data, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    return jsonify({'status': 'saved', 'filename': filename})
