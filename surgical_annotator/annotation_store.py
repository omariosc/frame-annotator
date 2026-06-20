"""Annotation storage system with auto-save.

Stores annotations as per-frame JSON files with automatic persistence.
Each frame gets its own file to prevent data loss from concurrent writes.

Directory layout::

    outputs/annotations/
    ├── 7DOF2024_..._Trial2/        # directory per trial (per-frame files)
    │   ├── frame_0000.json
    │   ├── frame_0100.json
    │   └── ...
    ├── 7DOF2024_..._Trial2.json    # old monolithic file (kept for compat)
    └── backups/                     # datetime-stamped backups
        └── 7DOF2024_..._Trial2_20260208_091500.json
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Output directory for annotations — uses AILET_DATA_DIR if set (macOS/portable),
# otherwise falls back to project-relative path (Windows default).
_data_dir_env = os.environ.get("AILET_DATA_DIR")
if _data_dir_env:
    ANNOTATION_DIR = Path(_data_dir_env) / 'outputs' / 'annotations'
else:
    ANNOTATION_DIR = Path(__file__).parent.parent.parent / 'outputs' / 'annotations'

# Sampling interval used for tool + peg progress counts. Mirrors
# FrameManager.SAMPLE_INTERVAL — duplicated here to avoid a circular import.
_PROGRESS_SAMPLE_INTERVAL = 100


@dataclass
class FrameAnnotation:
    """Annotations for a single frame.

    Structure includes mask, lines (top/bottom with auto-computed midline), and keypoints.
    Keypoints: joint, ee_tip, ee_left, ee_right for each tool.
    Visibility: 1=visible (requires annotation), 0=occluded (optional), -1=out of scene.
    """
    frame_idx: int
    tool1_mask: List[List[float]] = field(default_factory=list)  # Polygon vertices
    tool2_mask: List[List[float]] = field(default_factory=list)
    tool1_lines: Dict[str, List[List[float]]] = field(default_factory=lambda: {
        'top': [], 'bottom': []
    })
    tool2_lines: Dict[str, List[List[float]]] = field(default_factory=lambda: {
        'top': [], 'bottom': []
    })
    # Keypoints: [x, y] coordinates or empty list if not annotated
    tool1_joint: List[float] = field(default_factory=list)
    tool1_ee_tip: List[float] = field(default_factory=list)
    tool1_ee_left: List[float] = field(default_factory=list)
    tool1_ee_right: List[float] = field(default_factory=list)
    tool2_joint: List[float] = field(default_factory=list)
    tool2_ee_tip: List[float] = field(default_factory=list)
    tool2_ee_left: List[float] = field(default_factory=list)
    tool2_ee_right: List[float] = field(default_factory=list)
    # Visibility: 1=visible (requires annotation), 0=occluded (optional), -1=out of scene
    tool1_visibility: Dict[str, int] = field(default_factory=lambda: {
        'mask': 1, 'lines': 1, 'joint': 1, 'ee_tip': 1, 'ee_left': 1, 'ee_right': 1
    })
    tool2_visibility: Dict[str, int] = field(default_factory=lambda: {
        'mask': 1, 'lines': 1, 'joint': 1, 'ee_tip': 1, 'ee_left': 1, 'ee_right': 1
    })
    skipped: bool = False
    broken: bool = False     # Image moved to broken/ folder, unusable
    exclude: bool = False    # Usable for training only, excluded from validation
    last_modified: str = ''

    # --- Phase & Object Annotations (backward-compatible, default empty) ---
    pegs: List[Dict] = field(default_factory=list)
    # Each: {"id": 1, "bbox": [x,y,w,h], "mask": [[x,y]...], "state": "on_source_post",
    #        "post_id": null, "visible": true}

    pegboard: Dict = field(default_factory=lambda: {
        'source_posts': [],   # [[x,y]...] up to 6 centroids
        'target_posts': [],   # [[x,y]...] up to 6 centroids
        'source_post_masks': [[], [], [], [], [], []],  # polygon mask per source post
        'target_post_masks': [[], [], [], [], [], []],  # polygon mask per target post
        'source_post_keypoints': [None, None, None, None, None, None],  # center-top keypoint per source post
        'target_post_keypoints': [None, None, None, None, None, None],  # center-top keypoint per target post
        'board_mask': [],     # polygon outline
    })

    phase: Dict = field(default_factory=lambda: {
        'tool1': 'idle',
        'tool2': 'idle',
        'coarse': '',
        'fine': '',
        'cycle_index': 0,
        'active_tool': 1,
        'events': [],
    })

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'frame_idx': self.frame_idx,
            'tool1_mask': self.tool1_mask,
            'tool2_mask': self.tool2_mask,
            'tool1_lines': self.tool1_lines,
            'tool2_lines': self.tool2_lines,
            'tool1_joint': self.tool1_joint,
            'tool1_ee_tip': self.tool1_ee_tip,
            'tool1_ee_left': self.tool1_ee_left,
            'tool1_ee_right': self.tool1_ee_right,
            'tool2_joint': self.tool2_joint,
            'tool2_ee_tip': self.tool2_ee_tip,
            'tool2_ee_left': self.tool2_ee_left,
            'tool2_ee_right': self.tool2_ee_right,
            'tool1_visibility': self.tool1_visibility,
            'tool2_visibility': self.tool2_visibility,
            'skipped': self.skipped,
            'broken': self.broken,
            'exclude': self.exclude,
            'last_modified': self.last_modified,
            'pegs': self.pegs,
            'pegboard': self.pegboard,
            'phase': self.phase,
        }

    @classmethod
    def _migrate_visibility(cls, vis_data: Optional[Dict]) -> Dict[str, int]:
        """Migrate old visibility formats to full format with keypoints.

        Args:
            vis_data: Old visibility dict (may have missing keys, etc.)

        Returns:
            Full visibility dict with all keys:
            {mask, lines, joint, ee_tip, ee_left, ee_right}
            where 1=visible (requires annotation), 0=occluded (optional), -1=out of scene.
        """
        default = {
            'mask': 1, 'lines': 1,
            'joint': 1, 'ee_tip': 1, 'ee_left': 1, 'ee_right': 1
        }
        if not vis_data:
            return default

        # Preserve existing values, add missing keys with default 1
        result = {}
        for key in default:
            result[key] = vis_data.get(key, 1)

        return result

    @classmethod
    def _ensure_lines(cls, lines_data: Optional[Dict]) -> Dict[str, List]:
        """Ensure lines dict has only top/bottom keys."""
        default = {'top': [], 'bottom': []}
        if not lines_data:
            return default
        return {
            'top': lines_data.get('top', []),
            'bottom': lines_data.get('bottom', [])
        }

    @classmethod
    def _migrate_phase(cls, phase_data: Optional[Dict]) -> Dict:
        """Migrate legacy single-tool `coarse` phase to per-tool tool1/tool2."""
        default = {
            'tool1': 'idle', 'tool2': 'idle',
            'coarse': '', 'fine': '', 'cycle_index': 0,
            'active_tool': 1, 'events': [],
        }
        if not phase_data or not isinstance(phase_data, dict):
            return default
        result = dict(default)
        result.update({k: v for k, v in phase_data.items() if k in default})
        # If new schema fields missing but legacy `coarse` present, migrate.
        has_new = ('tool1' in phase_data) or ('tool2' in phase_data)
        if not has_new:
            coarse = phase_data.get('coarse', '') or ''
            at = phase_data.get('active_tool', 0)
            if coarse:
                if at == 2:
                    result['tool1'] = 'idle'
                    result['tool2'] = coarse
                else:
                    result['tool1'] = coarse
                    result['tool2'] = 'idle'
        # active_tool: 0 = both, 1 = tool1, 2 = tool2
        if result.get('active_tool') not in (0, 1, 2):
            result['active_tool'] = 0
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'FrameAnnotation':
        """Create from dictionary with backward-compatible migration.

        Migrates old visibility format and preserves keypoint fields.
        """
        # Migrate visibility
        t1_vis = cls._migrate_visibility(data.get('tool1_visibility') or data.get('tool1_missing'))
        t2_vis = cls._migrate_visibility(data.get('tool2_visibility') or data.get('tool2_missing'))

        return cls(
            frame_idx=data.get('frame_idx', 0),
            tool1_mask=data.get('tool1_mask', []),
            tool2_mask=data.get('tool2_mask', []),
            tool1_lines=cls._ensure_lines(data.get('tool1_lines')),
            tool2_lines=cls._ensure_lines(data.get('tool2_lines')),
            tool1_joint=data.get('tool1_joint', []),
            tool1_ee_tip=data.get('tool1_ee_tip', []),
            tool1_ee_left=data.get('tool1_ee_left', []),
            tool1_ee_right=data.get('tool1_ee_right', []),
            tool2_joint=data.get('tool2_joint', []),
            tool2_ee_tip=data.get('tool2_ee_tip', []),
            tool2_ee_left=data.get('tool2_ee_left', []),
            tool2_ee_right=data.get('tool2_ee_right', []),
            tool1_visibility=t1_vis,
            tool2_visibility=t2_vis,
            skipped=data.get('skipped', False),
            broken=data.get('broken', False),
            exclude=data.get('exclude', False),
            last_modified=data.get('last_modified', ''),
            pegs=data.get('pegs', []),
            pegboard={
                'source_posts': data.get('pegboard', {}).get('source_posts', []),
                'target_posts': data.get('pegboard', {}).get('target_posts', []),
                'source_post_masks': data.get('pegboard', {}).get('source_post_masks', [[], [], [], [], [], []]),
                'target_post_masks': data.get('pegboard', {}).get('target_post_masks', [[], [], [], [], [], []]),
                'source_post_keypoints': data.get('pegboard', {}).get('source_post_keypoints', [None, None, None, None, None, None]),
                'target_post_keypoints': data.get('pegboard', {}).get('target_post_keypoints', [None, None, None, None, None, None]),
                'board_mask': data.get('pegboard', {}).get('board_mask', []),
            },
            phase=cls._migrate_phase(data.get('phase')),
        )

    def is_peg_complete(self) -> bool:
        """Check if all peg + pegboard annotations are complete for this frame.

        Complete = all 6 P1–P6 pegs have a polygon mask (≥3 vertices) and
        3 keypoints, all 6 source posts AND all 6 target posts have a mask
        (≥3 vertices) and a centre keypoint, and the board outline polygon
        is drawn (≥3 vertices). Matches the strictest checklist shown in
        the pegs-mode status panel.
        """
        pegs = self.pegs or []
        for i in range(1, 7):
            peg = next((p for p in pegs if isinstance(p, dict) and p.get('id') == i), None)
            if not peg:
                return False
            if len(peg.get('mask') or []) < 3:
                return False
            kps = peg.get('keypoints') or []
            if len([k for k in kps if k is not None]) != 3:
                return False

        pb = self.pegboard if isinstance(self.pegboard, dict) else {}
        sp_masks = pb.get('source_post_masks') or [[], [], [], [], [], []]
        sp_kps = pb.get('source_post_keypoints') or [None] * 6
        tp_masks = pb.get('target_post_masks') or [[], [], [], [], [], []]
        tp_kps = pb.get('target_post_keypoints') or [None] * 6
        for i in range(6):
            if len(sp_masks[i] or []) < 3 or sp_kps[i] is None:
                return False
            if len(tp_masks[i] or []) < 3 or tp_kps[i] is None:
                return False

        if len(pb.get('board_mask') or []) < 3:
            return False
        return True

    def is_phase_annotated(self) -> bool:
        """Check if frame has a non-default phase label.

        A frame is phase-annotated when either per-tool field is non-idle,
        or the legacy ``coarse`` string is non-empty. Mirrors the inclusion
        rule used by ``/trials/<id>/phase_summary``.
        """
        p = self.phase if isinstance(self.phase, dict) else {}
        t1 = p.get('tool1') or 'idle'
        t2 = p.get('tool2') or 'idle'
        coarse = p.get('coarse', '') or ''
        return (t1 != 'idle') or (t2 != 'idle') or bool(coarse)

    def is_negative(self) -> bool:
        """Check if frame is negative (both tools entirely out of scene).

        A negative frame has all 6 visibility components set to -1 for both
        tools. These frames are valid annotations but don't represent real
        annotation work.
        """
        _VIS_KEYS = ('mask', 'lines', 'joint', 'ee_tip', 'ee_left', 'ee_right')
        t1v = self.tool1_visibility if isinstance(self.tool1_visibility, dict) else {}
        t2v = self.tool2_visibility if isinstance(self.tool2_visibility, dict) else {}
        t1_all_out = all(t1v.get(k, 1) == -1 for k in _VIS_KEYS)
        t2_all_out = all(t2v.get(k, 1) == -1 for k in _VIS_KEYS)
        return t1_all_out and t2_all_out

    def is_complete(self) -> bool:
        """Check if all annotations are complete for this frame.

        Visibility: 1=visible (requires annotation), 0=occluded (optional), -1=out of scene.
        A component is complete if: not visible (vis != 1) OR has valid annotation data.
        """
        if self.skipped or self.broken:
            return True

        t1v = self.tool1_visibility if isinstance(self.tool1_visibility, dict) else {}
        t2v = self.tool2_visibility if isinstance(self.tool2_visibility, dict) else {}

        # Helper: check if component needs annotation (visibility == 1)
        def needs_annotation(vis_val: int) -> bool:
            return vis_val == 1

        # Tool 1: visible (1) requires annotation, occluded (0) or out (-1) is excused
        if needs_annotation(t1v.get('mask', 1)) and not self.tool1_mask:
            return False
        if needs_annotation(t1v.get('lines', 1)):
            if len(self.tool1_lines.get('top', [])) != 2:
                return False
            if len(self.tool1_lines.get('bottom', [])) != 2:
                return False
        # Tool 1 keypoints
        if needs_annotation(t1v.get('joint', 1)) and len(self.tool1_joint) != 2:
            return False
        if needs_annotation(t1v.get('ee_tip', 1)) and len(self.tool1_ee_tip) != 2:
            return False
        if needs_annotation(t1v.get('ee_left', 1)) and len(self.tool1_ee_left) != 2:
            return False
        if needs_annotation(t1v.get('ee_right', 1)) and len(self.tool1_ee_right) != 2:
            return False

        # Tool 2: same logic
        if needs_annotation(t2v.get('mask', 1)) and not self.tool2_mask:
            return False
        if needs_annotation(t2v.get('lines', 1)):
            if len(self.tool2_lines.get('top', [])) != 2:
                return False
            if len(self.tool2_lines.get('bottom', [])) != 2:
                return False
        # Tool 2 keypoints
        if needs_annotation(t2v.get('joint', 1)) and len(self.tool2_joint) != 2:
            return False
        if needs_annotation(t2v.get('ee_tip', 1)) and len(self.tool2_ee_tip) != 2:
            return False
        if needs_annotation(t2v.get('ee_left', 1)) and len(self.tool2_ee_left) != 2:
            return False
        if needs_annotation(t2v.get('ee_right', 1)) and len(self.tool2_ee_right) != 2:
            return False

        return True


class AnnotationStore:
    """Manages annotation storage and persistence.

    Uses per-frame JSON files to prevent data loss. Each frame's annotation
    is stored in its own file under a trial directory. Falls back to reading
    monolithic JSON files for backward compatibility.
    """

    def __init__(self):
        self.annotations: Dict[str, Dict[int, FrameAnnotation]] = {}
        self._progress_cache: Dict[str, Dict[str, int]] = {}
        ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    def _get_store_path(self, trial_id: str) -> Path:
        """Get monolithic storage path for a trial's annotations."""
        safe_name = trial_id.replace('/', '_').replace('\\', '_')
        return ANNOTATION_DIR / f"{safe_name}.json"

    def _get_trial_dir(self, trial_id: str) -> Path:
        """Get per-frame directory path for a trial.

        Args:
            trial_id: Trial identifier.

        Returns:
            Path to trial's per-frame directory.
        """
        safe_name = trial_id.replace('/', '_').replace('\\', '_')
        return ANNOTATION_DIR / safe_name

    def _get_frame_path(self, trial_id: str, frame_idx: int) -> Path:
        """Get file path for a single frame's annotation.

        Args:
            trial_id: Trial identifier.
            frame_idx: Frame index.

        Returns:
            Path to frame JSON file.
        """
        return self._get_trial_dir(trial_id) / f"frame_{frame_idx:04d}.json"

    def save_frame(self, trial_id: str, frame_idx: int) -> bool:
        """Save a single frame's annotation to its own file.

        Args:
            trial_id: Trial identifier.
            frame_idx: Frame index.

        Returns:
            True if successful.
        """
        if trial_id not in self.annotations:
            return False

        frames = self.annotations[trial_id]
        if frame_idx not in frames:
            return False

        trial_dir = self._get_trial_dir(trial_id)
        trial_dir.mkdir(parents=True, exist_ok=True)

        frame_path = self._get_frame_path(trial_id, frame_idx)

        try:
            data = frames[frame_idx].to_dict()
            temp_path = frame_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            temp_path.replace(frame_path)
            self._save_progress_summary(trial_id)
            return True
        except Exception as e:
            logger.error(f"Error saving frame {frame_idx} for {trial_id}: {e}")
            return False

    def load_trial(self, trial_id: str) -> Dict[int, FrameAnnotation]:
        """Load annotations for a trial from disk.

        Tries per-frame directory first, falls back to monolithic JSON.
        If only monolithic exists, migrates to per-frame files.

        Args:
            trial_id: Trial identifier.

        Returns:
            Dict mapping frame index to FrameAnnotation.
        """
        if trial_id in self.annotations:
            return self.annotations[trial_id]

        trial_dir = self._get_trial_dir(trial_id)
        store_path = self._get_store_path(trial_id)

        frames: Dict[int, FrameAnnotation] = {}

        # Try per-frame directory first
        if trial_dir.is_dir():
            for frame_file in trial_dir.glob("frame_*.json"):
                try:
                    with open(frame_file, 'r') as f:
                        data = json.load(f)
                    frame_idx = data.get('frame_idx', 0)
                    frames[frame_idx] = FrameAnnotation.from_dict(data)
                except Exception as e:
                    logger.error(f"Error loading {frame_file}: {e}")

        # If no per-frame files, fall back to monolithic
        if not frames and store_path.exists():
            try:
                with open(store_path, 'r') as f:
                    data = json.load(f)

                for frame_str, frame_data in data.get('frames', {}).items():
                    frame_idx = int(frame_str)
                    frames[frame_idx] = FrameAnnotation.from_dict(frame_data)

                # Migrate: write per-frame files from monolithic
                if frames:
                    logger.info(
                        f"Migrating {len(frames)} frames from monolithic "
                        f"to per-frame for {trial_id}"
                    )
                    self.annotations[trial_id] = frames
                    trial_dir.mkdir(parents=True, exist_ok=True)
                    for frame_idx in frames:
                        self.save_frame(trial_id, frame_idx)

            except Exception as e:
                logger.error(f"Error loading annotations for {trial_id}: {e}")

        self.annotations[trial_id] = frames
        return frames

    def load_trial_sampled(
        self, trial_id: str, sampled_frames: set[int]
    ) -> Dict[int, FrameAnnotation]:
        """Load annotations for only the sampled frames of a trial.

        Instead of globbing ALL frame_*.json files (which can be 600+),
        directly opens only the files on the sampling grid. This prevents
        the UI freeze when trials have hundreds of off-grid batch files.

        Args:
            trial_id: Trial identifier.
            sampled_frames: Set of frame indices to load (e.g. {0, 100, 200, ...}).

        Returns:
            Dict mapping frame index to FrameAnnotation.
        """
        if trial_id in self.annotations:
            return self.annotations[trial_id]

        trial_dir = self._get_trial_dir(trial_id)
        store_path = self._get_store_path(trial_id)
        frames: Dict[int, FrameAnnotation] = {}

        if trial_dir.is_dir():
            for idx in sampled_frames:
                path = trial_dir / f"frame_{idx:04d}.json"
                if path.exists():
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                        frames[idx] = FrameAnnotation.from_dict(data)
                    except Exception as e:
                        logger.error(f"Error loading {path}: {e}")

        # Monolithic fallback (same as load_trial)
        if not frames and store_path.exists():
            try:
                with open(store_path, 'r') as f:
                    data = json.load(f)

                for frame_str, frame_data in data.get('frames', {}).items():
                    frame_idx = int(frame_str)
                    if frame_idx in sampled_frames:
                        frames[frame_idx] = FrameAnnotation.from_dict(frame_data)

                # Migrate: write per-frame files from monolithic
                if frames:
                    logger.info(
                        f"Migrating {len(frames)} sampled frames from "
                        f"monolithic to per-frame for {trial_id}"
                    )
                    self.annotations[trial_id] = frames
                    trial_dir.mkdir(parents=True, exist_ok=True)
                    for frame_idx in frames:
                        self.save_frame(trial_id, frame_idx)

            except Exception as e:
                logger.error(f"Error loading annotations for {trial_id}: {e}")

        self.annotations[trial_id] = frames
        return frames

    def save_trial(self, trial_id: str) -> bool:
        """Save all annotations for a trial as monolithic JSON.

        Used for batch operations and backups. Individual frame saves
        should use save_frame() instead.

        Args:
            trial_id: Trial identifier.

        Returns:
            True if successful.
        """
        if trial_id not in self.annotations:
            return False

        store_path = self._get_store_path(trial_id)

        try:
            frames_dict = {
                str(frame_idx): ann.to_dict()
                for frame_idx, ann in self.annotations[trial_id].items()
            }

            data = {
                'trial_id': trial_id,
                'last_saved': datetime.now().isoformat(),
                'frames': frames_dict
            }

            # Write to temp file first, then rename for atomic save
            temp_path = store_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            temp_path.replace(store_path)

            # Also write per-frame files (skip summary — we write it once below)
            for frame_idx in self.annotations[trial_id]:
                self._save_frame_file(trial_id, frame_idx)

            self._save_progress_summary(trial_id)
            return True
        except Exception as e:
            logger.error(f"Error saving annotations for {trial_id}: {e}")
            return False

    def _save_frame_file(self, trial_id: str, frame_idx: int) -> bool:
        """Write a single frame JSON without updating progress summary.

        Used by save_trial() to avoid writing the summary N times.

        Args:
            trial_id: Trial identifier.
            frame_idx: Frame index.

        Returns:
            True if successful.
        """
        frames = self.annotations.get(trial_id, {})
        if frame_idx not in frames:
            return False

        trial_dir = self._get_trial_dir(trial_id)
        trial_dir.mkdir(parents=True, exist_ok=True)

        frame_path = self._get_frame_path(trial_id, frame_idx)

        try:
            data = frames[frame_idx].to_dict()
            temp_path = frame_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            temp_path.replace(frame_path)
            return True
        except Exception as e:
            logger.error(f"Error saving frame file {frame_idx} for {trial_id}: {e}")
            return False

    def backup_trial(self, trial_id: str) -> Optional[Path]:
        """Create a datetime-stamped backup of all annotations for a trial.

        Args:
            trial_id: Trial identifier.

        Returns:
            Path to backup file, or None on failure.
        """
        if trial_id not in self.annotations:
            self.load_trial(trial_id)

        frames = self.annotations.get(trial_id, {})
        if not frames:
            return None

        backup_dir = ANNOTATION_DIR / 'backups'
        backup_dir.mkdir(parents=True, exist_ok=True)

        safe_name = trial_id.replace('/', '_').replace('\\', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f"{safe_name}_{timestamp}.json"

        try:
            frames_dict = {
                str(frame_idx): ann.to_dict()
                for frame_idx, ann in frames.items()
            }

            data = {
                'trial_id': trial_id,
                'backup_created': datetime.now().isoformat(),
                'frames': frames_dict
            }

            with open(backup_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Error creating backup for {trial_id}: {e}")
            return None

    def get_frame(self, trial_id: str, frame_idx: int) -> FrameAnnotation:
        """Get annotation for a specific frame.

        If the frame isn't in the in-memory cache (e.g. trial was loaded via
        ``load_trial_sampled``), checks disk before creating a blank annotation.
        This prevents blank annotations from silently overwriting real data.

        Args:
            trial_id: Trial identifier
            frame_idx: Frame index

        Returns:
            FrameAnnotation (loads from disk or creates new if not on disk)
        """
        if trial_id not in self.annotations:
            self.load_trial(trial_id)

        frames = self.annotations.get(trial_id, {})
        if frame_idx not in frames:
            # Check disk before creating blank — avoids overwriting real data
            # when trial was loaded with sampled-only frames
            frame_path = self._get_frame_path(trial_id, frame_idx)
            if frame_path.exists():
                try:
                    with open(frame_path) as f:
                        frames[frame_idx] = FrameAnnotation.from_dict(json.load(f))
                except Exception:
                    frames[frame_idx] = FrameAnnotation(frame_idx=frame_idx)
            else:
                frames[frame_idx] = FrameAnnotation(frame_idx=frame_idx)
            self.annotations[trial_id] = frames

        return frames[frame_idx]

    def get_single_frame(self, trial_id: str, frame_idx: int) -> FrameAnnotation:
        """Get annotation for a single frame without loading the full trial.

        If the trial is already cached, returns from cache (fast path).
        Otherwise reads just the one frame JSON file from disk.
        Does NOT populate self.annotations — that's for load_trial().

        Args:
            trial_id: Trial identifier.
            frame_idx: Frame index.

        Returns:
            FrameAnnotation for the requested frame.
        """
        # Fast path: trial already cached
        if trial_id in self.annotations:
            frames = self.annotations[trial_id]
            if frame_idx in frames:
                return frames[frame_idx]
            return FrameAnnotation(frame_idx=frame_idx)

        # Slow path: read single file from disk
        frame_path = self._get_frame_path(trial_id, frame_idx)
        if frame_path.exists():
            try:
                with open(frame_path, 'r') as f:
                    data = json.load(f)
                return FrameAnnotation.from_dict(data)
            except Exception as e:
                logger.error(f"Error loading single frame {frame_path}: {e}")

        return FrameAnnotation(frame_idx=frame_idx)

    def count_annotation_files(self, trial_id: str) -> int:
        """Count annotation files for a trial without reading them.

        Args:
            trial_id: Trial identifier.

        Returns:
            Number of frame_*.json files in the trial directory.
        """
        trial_dir = self._get_trial_dir(trial_id)
        if trial_dir.is_dir():
            return sum(1 for _ in trial_dir.glob("frame_*.json"))

        # Check monolithic file as fallback
        store_path = self._get_store_path(trial_id)
        if store_path.exists():
            try:
                with open(store_path, 'r') as f:
                    data = json.load(f)
                return len(data.get('frames', {}))
            except Exception:
                pass

        return 0

    def count_annotation_files_sampled(
        self, trial_id: str, sampled_frames: set[int]
    ) -> int:
        """Count annotation files that exist for sampled frames only.

        Args:
            trial_id: Trial identifier.
            sampled_frames: Set of frame indices on the sampling grid.

        Returns:
            Number of sampled frame_*.json files that exist on disk.
        """
        trial_dir = self._get_trial_dir(trial_id)
        if not trial_dir.is_dir():
            return 0
        return sum(
            1 for idx in sampled_frames
            if (trial_dir / f"frame_{idx:04d}.json").exists()
        )

    def load_trial_progressive(self, trial_id: str):
        """Load annotations progressively, yielding (loaded, total) after each file.

        If trial is already cached, yields (total, total) once and returns.
        Otherwise loads frame files one by one, yielding progress.
        At end, stores into self.annotations[trial_id].

        Args:
            trial_id: Trial identifier.

        Yields:
            Tuple of (loaded_count, total_count).
        """
        # Fast path: already cached
        if trial_id in self.annotations:
            total = len(self.annotations[trial_id])
            yield (total, total)
            return

        trial_dir = self._get_trial_dir(trial_id)
        store_path = self._get_store_path(trial_id)
        frames: Dict[int, FrameAnnotation] = {}

        # Try per-frame directory first
        if trial_dir.is_dir():
            frame_files = sorted(trial_dir.glob("frame_*.json"))
            total = len(frame_files)
            if total == 0:
                yield (0, 0)
            else:
                for i, frame_file in enumerate(frame_files):
                    try:
                        with open(frame_file, 'r') as f:
                            data = json.load(f)
                        frame_idx = data.get('frame_idx', 0)
                        frames[frame_idx] = FrameAnnotation.from_dict(data)
                    except Exception as e:
                        logger.error(f"Error loading {frame_file}: {e}")
                    yield (i + 1, total)

        # If no per-frame files, fall back to monolithic
        if not frames and store_path.exists():
            try:
                with open(store_path, 'r') as f:
                    data = json.load(f)

                items = list(data.get('frames', {}).items())
                total = len(items)
                for i, (frame_str, frame_data) in enumerate(items):
                    frame_idx = int(frame_str)
                    frames[frame_idx] = FrameAnnotation.from_dict(frame_data)
                    yield (i + 1, total)

                # Migrate: write per-frame files from monolithic
                if frames:
                    logger.info(
                        f"Migrating {len(frames)} frames from monolithic "
                        f"to per-frame for {trial_id}"
                    )
                    self.annotations[trial_id] = frames
                    trial_dir.mkdir(parents=True, exist_ok=True)
                    for frame_idx in frames:
                        self.save_frame(trial_id, frame_idx)

            except Exception as e:
                logger.error(f"Error loading annotations for {trial_id}: {e}")
                yield (0, 0)
        elif not frames:
            yield (0, 0)

        self.annotations[trial_id] = frames

    @staticmethod
    def _frame_idx_from_path(path: Path) -> int:
        """Extract frame index from a frame_NNNN.json filename.

        Args:
            path: Path to a frame JSON file.

        Returns:
            Frame index, or -1 if parsing fails.
        """
        try:
            return int(path.stem.split('_')[1])
        except (ValueError, IndexError):
            return -1

    def load_remaining_frames_progressive(self, trial_id: str):
        """Load annotation files NOT already in cache, yielding progress.

        Safe to call while user navigates — never clears existing cache entries.
        Only ADDS new frames from disk that aren't already cached.

        Args:
            trial_id: Trial identifier.

        Yields:
            Tuple of (loaded_count, total_count).
        """
        trial_dir = self._get_trial_dir(trial_id)
        if not trial_dir.is_dir():
            yield (0, 0)
            return

        existing = self.annotations.get(trial_id, {})
        frame_files = sorted(trial_dir.glob("frame_*.json"))
        # Filter to only files NOT already cached
        to_load = [
            f for f in frame_files
            if self._frame_idx_from_path(f) not in existing
        ]
        total = len(to_load)

        if total == 0:
            yield (0, 0)
            return

        for i, frame_file in enumerate(to_load):
            try:
                with open(frame_file, 'r') as f:
                    data = json.load(f)
                frame_idx = data.get('frame_idx', 0)
                existing[frame_idx] = FrameAnnotation.from_dict(data)
            except Exception as e:
                logger.error(f"Error loading {frame_file}: {e}")
            yield (i + 1, total)

        self.annotations[trial_id] = existing

    def load_trial_progressive_sampled(
        self, trial_id: str, sampled_frames: set[int]
    ):
        """Load only sampled-frame annotations progressively.

        Same yield pattern as load_trial_progressive but reads only files
        on the sampling grid instead of ALL frame_*.json files.

        Args:
            trial_id: Trial identifier.
            sampled_frames: Set of frame indices to load.

        Yields:
            Tuple of (loaded_count, total_count).
        """
        if trial_id in self.annotations:
            total = len(self.annotations[trial_id])
            yield (total, total)
            return

        trial_dir = self._get_trial_dir(trial_id)
        store_path = self._get_store_path(trial_id)
        frames: Dict[int, FrameAnnotation] = {}

        sorted_indices = sorted(sampled_frames)
        total = len(sorted_indices)

        if trial_dir.is_dir():
            for i, idx in enumerate(sorted_indices):
                path = trial_dir / f"frame_{idx:04d}.json"
                if path.exists():
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                        frames[idx] = FrameAnnotation.from_dict(data)
                    except Exception as e:
                        logger.error(f"Error loading {path}: {e}")
                yield (i + 1, total)

        # Monolithic fallback
        if not frames and store_path.exists():
            try:
                with open(store_path, 'r') as f:
                    data = json.load(f)

                all_items = data.get('frames', {})
                for i, idx in enumerate(sorted_indices):
                    frame_data = all_items.get(str(idx))
                    if frame_data:
                        frames[idx] = FrameAnnotation.from_dict(frame_data)
                    yield (i + 1, total)

                if frames:
                    logger.info(
                        f"Migrating {len(frames)} sampled frames from "
                        f"monolithic to per-frame for {trial_id}"
                    )
                    self.annotations[trial_id] = frames
                    trial_dir.mkdir(parents=True, exist_ok=True)
                    for frame_idx in frames:
                        self.save_frame(trial_id, frame_idx)

            except Exception as e:
                logger.error(f"Error loading annotations for {trial_id}: {e}")
                yield (0, 0)
        elif not frames:
            yield (0, 0)

        self.annotations[trial_id] = frames

    def update_frame(
        self,
        trial_id: str,
        frame_idx: int,
        updates: Dict[str, Any],
        auto_save: bool = True
    ) -> FrameAnnotation:
        """Update annotation for a frame.

        Args:
            trial_id: Trial identifier
            frame_idx: Frame index
            updates: Dict of fields to update
            auto_save: Whether to auto-save to disk

        Returns:
            Updated FrameAnnotation
        """
        frame = self.get_frame(trial_id, frame_idx)

        # Ensure keypoint fields exist before updating (defensive initialization)
        keypoint_fields = [
            'tool1_joint', 'tool1_ee_tip', 'tool1_ee_left', 'tool1_ee_right',
            'tool2_joint', 'tool2_ee_tip', 'tool2_ee_left', 'tool2_ee_right'
        ]
        for field in keypoint_fields:
            if getattr(frame, field, None) is None:
                setattr(frame, field, [])

        for key, value in updates.items():
            if hasattr(frame, key):
                if key == 'phase' and isinstance(value, dict):
                    # Preserve per-tool fields unless the caller explicitly set them.
                    merged = dict(frame.phase or {})
                    merged.update(value)
                    setattr(frame, key, FrameAnnotation._migrate_phase(merged))
                else:
                    setattr(frame, key, value)

        frame.last_modified = datetime.now().isoformat()

        if auto_save:
            self.save_frame(trial_id, frame_idx)

        return frame

    def skip_frame(self, trial_id: str, frame_idx: int) -> FrameAnnotation:
        """Mark a frame as skipped.

        Args:
            trial_id: Trial identifier
            frame_idx: Frame index

        Returns:
            Updated FrameAnnotation
        """
        return self.update_frame(trial_id, frame_idx, {'skipped': True})

    def get_prior_annotation(
        self,
        trial_id: str,
        current_frame: int,
        valid_frames: List[int]
    ) -> Optional[FrameAnnotation]:
        """Get the previous frame's annotation as a prior.

        Args:
            trial_id: Trial identifier
            current_frame: Current frame index
            valid_frames: List of valid frame indices

        Returns:
            Previous FrameAnnotation or None
        """
        if trial_id not in self.annotations:
            self.load_trial(trial_id)

        frames = self.annotations.get(trial_id, {})

        # Find previous frame in valid_frames
        prev_frame = None
        for frame_idx in valid_frames:
            if frame_idx >= current_frame:
                break
            prev_frame = frame_idx

        if prev_frame is not None and prev_frame in frames:
            return frames[prev_frame]

        return None

    def _save_progress_summary(self, trial_id: str) -> None:
        """Write lightweight progress summary for fast dashboard loading.

        Counts annotation statuses from in-memory data and writes a small
        ``_progress.json`` file in the trial's annotation directory. The
        progress endpoint reads these instead of loading all frame JSONs.

        Args:
            trial_id: Trial identifier.
        """
        frames = self.annotations.get(trial_id, {})
        if not frames:
            return

        counts: Dict[str, int] = {
            'completed': 0, 'skipped': 0, 'negative': 0, 'partial': 0, 'broken': 0,
            'peg_completed': 0,
            'phase_completed': 0,
            'broken_total': 0,
            'excluded_total': 0,
        }
        for frame_idx, ann in frames.items():
            # Full-trial counts: phase, broken_total, excluded_total
            if ann.broken:
                counts['broken_total'] += 1
            if ann.exclude:
                counts['excluded_total'] += 1
            # Phase coverage skips broken / excluded frames in both num and denom
            # (see CLAUDE.md). Count only in-scope phase-annotated frames here.
            if not ann.broken and not ann.exclude and ann.is_phase_annotated():
                counts['phase_completed'] += 1

            # Sampled-grid counts: tool + peg
            if frame_idx % _PROGRESS_SAMPLE_INTERVAL != 0:
                continue
            if ann.broken:
                counts['broken'] += 1
            elif ann.skipped:
                counts['skipped'] += 1
            elif ann.is_negative():
                counts['negative'] += 1
            elif ann.is_complete():
                counts['completed'] += 1
            else:
                counts['partial'] += 1
            if ann.is_peg_complete():
                counts['peg_completed'] += 1

        trial_dir = self._get_trial_dir(trial_id)
        trial_dir.mkdir(parents=True, exist_ok=True)
        summary_path = trial_dir / '_progress.json'
        try:
            temp_path = summary_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(counts, f)
            temp_path.replace(summary_path)
        except Exception as e:
            logger.warning(f"Failed to save progress summary for {trial_id}: {e}")
        self._progress_cache[trial_id] = counts

    def load_progress_summary(self, trial_id: str) -> Optional[Dict[str, int]]:
        """Read per-trial progress summary (cached in memory).

        Returns cached value if available, otherwise reads from disk and caches.

        Args:
            trial_id: Trial identifier.

        Returns:
            Dict with counts {completed, skipped, negative, partial, broken}
            or None if no summary file exists.
        """
        if trial_id in self._progress_cache:
            return self._progress_cache[trial_id]

        summary_path = self._get_trial_dir(trial_id) / '_progress.json'
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    data = json.load(f)
                self._progress_cache[trial_id] = data
                return data
            except Exception:
                pass
        return None

    def refresh_trial_progress_from_files(
        self, trial_id: str, sampled_frames: set[int]
    ) -> dict[str, int]:
        """Recompute progress for a trial by reading sampled frame files from disk.

        Bypasses the in-memory cache — reads directly from per-frame JSON files.
        Classifies each annotation and writes a fresh ``_progress.json`` atomically.

        Args:
            trial_id: Trial identifier.
            sampled_frames: Set of frame indices on the sampling grid.

        Returns:
            Dict with counts {completed, skipped, negative, partial, broken}.
        """
        trial_dir = self._get_trial_dir(trial_id)
        counts: dict[str, int] = {
            'completed': 0, 'skipped': 0, 'negative': 0,
            'partial': 0, 'broken': 0,
            'peg_completed': 0,
            'phase_completed': 0,
            'broken_total': 0,
            'excluded_total': 0,
        }
        if not trial_dir.is_dir():
            return counts

        # Single full-directory pass: classify every existing frame file
        # for the full-trial counts (phase / broken_total / excluded_total),
        # and additionally tally tool + peg slots when on the sampling grid.
        for path in trial_dir.glob("frame_*.json"):
            try:
                idx = int(path.stem.split('_')[1])
            except (ValueError, IndexError):
                continue
            try:
                with open(path) as f:
                    ann = FrameAnnotation.from_dict(json.load(f))
            except Exception as e:
                logger.warning(f"Error reading {path}: {e}")
                continue

            if ann.broken:
                counts['broken_total'] += 1
            if ann.exclude:
                counts['excluded_total'] += 1
            # Phase coverage skips broken / excluded frames in both num and denom
            # (see CLAUDE.md). Count only in-scope phase-annotated frames here.
            if not ann.broken and not ann.exclude and ann.is_phase_annotated():
                counts['phase_completed'] += 1

            if idx not in sampled_frames:
                continue
            if ann.broken:
                counts['broken'] += 1
            elif ann.skipped:
                counts['skipped'] += 1
            elif ann.is_negative():
                counts['negative'] += 1
            elif ann.is_complete():
                counts['completed'] += 1
            else:
                counts['partial'] += 1
            if ann.is_peg_complete():
                counts['peg_completed'] += 1

        # Write fresh _progress.json atomically
        trial_dir.mkdir(parents=True, exist_ok=True)
        summary_path = trial_dir / '_progress.json'
        try:
            temp_path = summary_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(counts, f)
            temp_path.replace(summary_path)
        except Exception as e:
            logger.warning(f"Failed to write progress summary for {trial_id}: {e}")
        self._progress_cache[trial_id] = counts

        return counts

    def seed_progress_summaries(self, sample_interval: int = 100) -> int:
        """Generate _progress.json for all existing trial annotation dirs.

        Scans the annotation directory for trial folders, loads only their
        sampled frame files (idx % sample_interval == 0), computes counts,
        and writes summaries. Off-grid batch files are ignored so that
        progress percentages reflect only the annotation sampling grid.

        Args:
            sample_interval: Frame sampling interval (default 100).

        Returns:
            Number of summaries written.
        """
        count = 0
        for trial_dir in ANNOTATION_DIR.iterdir():
            if not trial_dir.is_dir() or trial_dir.name == 'backups':
                continue

            all_files = list(trial_dir.glob("frame_*.json"))
            if not all_files:
                continue

            counts: Dict[str, int] = {
                'completed': 0, 'skipped': 0, 'negative': 0,
                'partial': 0, 'broken': 0,
                'peg_completed': 0,
                'phase_completed': 0,
                'broken_total': 0,
                'excluded_total': 0,
            }
            for frame_file in all_files:
                try:
                    idx = int(frame_file.stem.split('_')[1])
                except (ValueError, IndexError):
                    continue
                try:
                    with open(frame_file, 'r') as f:
                        data = json.load(f)
                    ann = FrameAnnotation.from_dict(data)
                except Exception as e:
                    logger.warning(f"Error reading {frame_file}: {e}")
                    continue

                if ann.broken:
                    counts['broken_total'] += 1
                if ann.exclude:
                    counts['excluded_total'] += 1
                # Phase coverage skips broken / excluded frames in both num
                # and denom (see CLAUDE.md).
                if not ann.broken and not ann.exclude and ann.is_phase_annotated():
                    counts['phase_completed'] += 1

                if idx % sample_interval != 0:
                    continue
                if ann.broken:
                    counts['broken'] += 1
                elif ann.skipped:
                    counts['skipped'] += 1
                elif ann.is_negative():
                    counts['negative'] += 1
                elif ann.is_complete():
                    counts['completed'] += 1
                else:
                    counts['partial'] += 1
                if ann.is_peg_complete():
                    counts['peg_completed'] += 1

            summary_path = trial_dir / '_progress.json'
            try:
                with open(summary_path, 'w') as f:
                    json.dump(counts, f)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to write summary for {trial_dir.name}: {e}")

        logger.info(f"Seeded {count} progress summaries (interval={sample_interval})")
        return count

    def warmup_progress_cache(self, fm) -> None:
        """Pre-load all _progress.json files into memory.

        Args:
            fm: FrameManager instance (used to enumerate all trials).
        """
        for dataset_name, trials in fm.discover_all_trials().items():
            for trial_name in trials:
                self.load_progress_summary(f"{dataset_name}/{trial_name}")

    def get_trial_progress(self, trial_id: str, valid_frames: List[int]) -> Dict:
        """Get annotation progress for a trial.

        Args:
            trial_id: Trial identifier
            valid_frames: List of valid frame indices

        Returns:
            Progress info dict
        """
        if trial_id not in self.annotations:
            self.load_trial(trial_id)

        frames = self.annotations.get(trial_id, {})

        completed = 0
        skipped = 0
        broken = 0
        excluded = 0
        partial = 0
        negative = 0

        for frame_idx in valid_frames:
            ann = frames.get(frame_idx)
            if ann:
                if ann.broken:
                    broken += 1
                elif ann.skipped:
                    skipped += 1
                elif ann.is_negative():
                    negative += 1
                elif ann.is_complete():
                    completed += 1
                else:
                    partial += 1
                if ann.exclude:
                    excluded += 1

        # Count completed/skipped off-sample frames (e.g. batch-annotated)
        valid_set = set(valid_frames)
        extra_completed = 0
        extra_skipped = 0
        extra_negative = 0
        for frame_idx, ann in frames.items():
            if frame_idx not in valid_set:
                if ann.broken:
                    broken += 1
                elif ann.skipped:
                    extra_skipped += 1
                elif ann.is_negative():
                    extra_negative += 1
                elif ann.is_complete():
                    extra_completed += 1
                if ann.exclude:
                    excluded += 1

        total = len(valid_frames) + extra_completed + extra_skipped + extra_negative
        completed += extra_completed
        skipped += extra_skipped
        negative += extra_negative

        return {
            'total': total,
            'completed': completed,
            'skipped': skipped,
            'broken': broken,
            'excluded': excluded,
            'partial': partial,
            'negative': negative,
            'remaining': total - completed - skipped - broken - negative
        }


# Global annotation store instance
_annotation_store: Optional[AnnotationStore] = None


def get_annotation_store() -> AnnotationStore:
    """Get or create the global annotation store instance."""
    global _annotation_store
    if _annotation_store is None:
        _annotation_store = AnnotationStore()
    return _annotation_store


def load_annotations_any_format(
    annotation_dir: Path,
    trial_id: str,
) -> Optional[Dict[str, Any]]:
    """Load annotations for a trial from any storage format.

    Tries per-frame directory first, then falls back to monolithic JSON.
    Returns data in the monolithic format (with 'frames' dict) for
    compatibility with consumers like export_yolo and multimodal_loader.

    Args:
        annotation_dir: Base annotations directory.
        trial_id: Trial identifier (e.g. "7DOF2024/Attempt 1/Trial1").

    Returns:
        Dict with 'trial_id' and 'frames' keys, or None if no data found.
    """
    safe_name = trial_id.replace('/', '_').replace('\\', '_')
    trial_dir = annotation_dir / safe_name
    monolithic_path = annotation_dir / f"{safe_name}.json"

    frames: Dict[str, Any] = {}

    # Try per-frame directory first
    if trial_dir.is_dir():
        for frame_file in sorted(trial_dir.glob("frame_*.json")):
            try:
                with open(frame_file, 'r') as f:
                    data = json.load(f)
                frame_idx = data.get('frame_idx', 0)
                frames[str(frame_idx)] = data
            except Exception as e:
                logger.error(f"Error loading {frame_file}: {e}")

    # Fall back to monolithic if no per-frame files found
    if not frames and monolithic_path.exists():
        try:
            with open(monolithic_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading {monolithic_path}: {e}")
            return None

    if not frames:
        return None

    return {
        'trial_id': trial_id,
        'frames': frames,
    }
