"""Frame manager for discovering and sampling frames across datasets.

Handles frame discovery, NaN detection, and navigation for annotation tool.
"""

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Import base paths
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from surgical_annotator.config import DATA_6DOF, DATA_7DOF, DATA_BAPES


@dataclass
class TrialInfo:
    """Information about a trial."""
    dataset: str
    trial_name: str
    trial_path: Path
    frames_dir: Path
    label_path: Path
    frame_pattern: str  # e.g., "frame_{:05d}.bmp" or "test1_{:04d}.png"
    total_frames: int = 0
    sampled_frames: List[int] = field(default_factory=list)
    valid_frames: Optional[List[int]] = None  # After NaN filtering; None = not computed


class FrameManager:
    """Manages frame discovery and sampling across all datasets."""

    SAMPLE_INTERVAL = 100  # Sample every 100 frames

    def __init__(self):
        self.trials: Dict[str, TrialInfo] = {}
        self.datasets = {
            '6DOF2023': DATA_6DOF,
            '7DOF2024': DATA_7DOF,
            'BAPES2024': DATA_BAPES
        }
        self._kinematics_cache: Dict[str, Dict] = {}
        self._all_trials_cache: Optional[Dict[str, List[str]]] = None

    def discover_all_trials(self) -> Dict[str, List[str]]:
        """Discover all trials across all datasets.

        Returns cached result on subsequent calls.

        Returns:
            Dict mapping dataset name to list of trial names
        """
        if self._all_trials_cache is not None:
            return self._all_trials_cache

        result = {}

        for dataset_name, dataset_path in self.datasets.items():
            if not dataset_path.exists():
                continue

            trials = []
            if dataset_name == '6DOF2023':
                trials = self._discover_6dof_trials(dataset_path)
            elif dataset_name == '7DOF2024':
                trials = self._discover_7dof_trials(dataset_path)
            elif dataset_name == 'BAPES2024':
                trials = self._discover_bapes_trials(dataset_path)

            result[dataset_name] = trials

        self._all_trials_cache = result
        return result

    def warmup(self) -> None:
        """Pre-compute all trial metadata: discovery, frame counts."""
        self.discover_all_trials()
        for trial_info in self.trials.values():
            self._ensure_frame_count(trial_info)

    def _discover_6dof_trials(self, dataset_path: Path) -> List[str]:
        """Discover trials in 6DOF2023 dataset.

        Prefers PNG directories ("Test X png") since they have sequential naming.
        Falls back to BMP directories ("Test X") if no PNG version exists.
        Does NOT require label.json — 6DOF2023 has no label files.
        """
        trials = []
        # Track which test numbers have PNG versions
        png_test_nums = set()

        # First pass: discover PNG directories
        for item in sorted(dataset_path.iterdir()):
            if not item.is_dir():
                continue

            trial_name = item.name

            if 'png' in trial_name.lower():
                base_name = trial_name.replace(' png', '').replace(' PNG', '')
                has_files = any(item.glob('*.png'))
                if has_files:
                    test_num = ''.join(filter(str.isdigit, base_name))
                    png_test_nums.add(test_num)
                    trial_id = f"6DOF2023/{trial_name}"
                    # label.json may not exist for 6DOF — that's OK
                    label_path = dataset_path / base_name / 'label.json'
                    self.trials[trial_id] = TrialInfo(
                        dataset='6DOF2023',
                        trial_name=trial_name,
                        trial_path=item,
                        frames_dir=item,
                        label_path=label_path,
                        frame_pattern=f"test{test_num}_{{:04d}}.png",
                        total_frames=0  # counted lazily
                    )
                    trials.append(trial_name)

        # Second pass: discover BMP directories only if no PNG version exists
        for item in sorted(dataset_path.iterdir()):
            if not item.is_dir():
                continue

            trial_name = item.name
            if 'png' in trial_name.lower() or 'txt' in trial_name.lower():
                continue
            if not trial_name.startswith('Test'):
                continue

            test_num = ''.join(filter(str.isdigit, trial_name))
            if test_num in png_test_nums:
                continue  # Already have PNG version

            has_bmp = any(item.glob('*.bmp'))
            if has_bmp:
                label_path = item / 'label.json'
                trial_id = f"6DOF2023/{trial_name}"
                self.trials[trial_id] = TrialInfo(
                    dataset='6DOF2023',
                    trial_name=trial_name,
                    trial_path=item,
                    frames_dir=item,
                    label_path=label_path,
                    frame_pattern="{:d}.bmp",
                    total_frames=0  # counted lazily
                )
                trials.append(trial_name)

        return trials

    def _discover_7dof_trials(self, dataset_path: Path) -> List[str]:
        """Discover trials in 7DOF2024 dataset.

        Handles both top-level Trial folders and trials inside Attempt folders.
        """
        trials = []

        for item in sorted(dataset_path.iterdir()):
            if not item.is_dir():
                continue

            # Check if this is an Attempt folder containing trials
            if item.name.startswith('Attempt'):
                # Look for Trial folders inside Attempt folder
                for trial_item in sorted(item.iterdir()):
                    if not trial_item.is_dir() or not trial_item.name.startswith('Trial'):
                        continue

                    frames_dir = trial_item / 'Frames'
                    label_path = trial_item / 'label.json'

                    if frames_dir.exists() and label_path.exists():
                        trial_name = f"{item.name}/{trial_item.name}"
                        trial_id = f"7DOF2024/{trial_name}"
                        self.trials[trial_id] = TrialInfo(
                            dataset='7DOF2024',
                            trial_name=trial_name,
                            trial_path=trial_item,
                            frames_dir=frames_dir,
                            label_path=label_path,
                            frame_pattern="frame_{:05d}.bmp",
                            total_frames=0  # counted lazily
                        )
                        trials.append(trial_name)

            # Check if this is a top-level Trial folder
            elif item.name.startswith('Trial'):
                frames_dir = item / 'Frames'
                label_path = item / 'label.json'

                if frames_dir.exists() and label_path.exists():
                    trial_id = f"7DOF2024/{item.name}"
                    self.trials[trial_id] = TrialInfo(
                        dataset='7DOF2024',
                        trial_name=item.name,
                        trial_path=item,
                        frames_dir=frames_dir,
                        label_path=label_path,
                        frame_pattern="frame_{:05d}.bmp",
                        total_frames=0  # counted lazily
                    )
                    trials.append(item.name)

        return trials

    def _discover_bapes_trials(self, dataset_path: Path) -> List[str]:
        """Discover trials in BAPES2024 dataset."""
        trials = []

        # BAPES has subdirectories: Industry and MIS Course
        for subdir in ['Industry', 'MIS Course']:
            subdir_path = dataset_path / subdir
            if not subdir_path.exists():
                continue

            for item in sorted(subdir_path.iterdir()):
                if not item.is_dir() or not item.name.startswith('Trial'):
                    continue

                frames_dir = item / 'Frames'
                label_path = item / 'label.json'

                if frames_dir.exists() and label_path.exists():
                    trial_name = f"{subdir}/{item.name}"
                    trial_id = f"BAPES2024/{trial_name}"
                    self.trials[trial_id] = TrialInfo(
                        dataset='BAPES2024',
                        trial_name=trial_name,
                        trial_path=item,
                        frames_dir=frames_dir,
                        label_path=label_path,
                        frame_pattern="frame_{:05d}.bmp",
                        total_frames=0  # counted lazily
                    )
                    trials.append(trial_name)

        return trials

    def load_kinematics(self, trial_id: str) -> Optional[Dict]:
        """Load kinematics data for a trial.

        Args:
            trial_id: Trial identifier (e.g., "7DOF2024/Trial1")

        Returns:
            Kinematics data dict or None if not found
        """
        if trial_id in self._kinematics_cache:
            return self._kinematics_cache[trial_id]

        trial = self.trials.get(trial_id)
        if not trial or not trial.label_path.exists():
            return None

        try:
            with open(trial.label_path, 'r') as f:
                content = f.read()

            # Fix trailing commas in JSON (common issue in some label files)
            content = re.sub(r',\s*]', ']', content)
            content = re.sub(r',\s*}', '}', content)

            data = json.loads(content)
            self._kinematics_cache[trial_id] = data
            return data
        except Exception as e:
            print(f"Error loading kinematics for {trial_id}: {e}")
            return None

    def _ensure_frame_count(self, trial: TrialInfo) -> int:
        """Compute total frame span on first access using os.scandir.

        Sets total_frames to (max_index + 1), so sampling covers the full
        index range even when some files have been moved to broken/.

        Args:
            trial: TrialInfo whose total_frames may be 0 (not yet counted).

        Returns:
            The total frame span (max_index + 1).
        """
        if trial.total_frames > 0:
            return trial.total_frames

        suffix = '.png' if trial.frame_pattern.endswith('.png') else '.bmp'
        max_idx = -1
        try:
            for entry in os.scandir(trial.frames_dir):
                if not entry.is_file(follow_symlinks=False):
                    continue
                if not entry.name.endswith(suffix):
                    continue
                name = entry.name
                if trial.dataset == '6DOF2023':
                    if suffix == '.png':
                        # test1_0000.png → extract digits after last _
                        parts = name.rsplit('_', 1)
                        if len(parts) == 2:
                            try:
                                idx = int(parts[1].split('.')[0])
                                max_idx = max(max_idx, idx)
                            except ValueError:
                                pass
                    else:
                        # 0.bmp, 1.bmp → the number IS the filename
                        try:
                            idx = int(name.split('.')[0])
                            max_idx = max(max_idx, idx)
                        except ValueError:
                            pass
                else:
                    # frame_00050.bmp → extract digits after frame_
                    if name.startswith('frame_'):
                        try:
                            idx = int(name[6:].split('.')[0])
                            max_idx = max(max_idx, idx)
                        except ValueError:
                            pass
        except OSError:
            max_idx = -1

        trial.total_frames = max_idx + 1 if max_idx >= 0 else 0
        return trial.total_frames

    def has_valid_kinematics(self, annotation: Dict) -> bool:
        """Check if a frame annotation has valid kinematics (no NaNs).

        Tool 1, Tool 2, and World must have valid positions/rotations.
        Camera NaN is acceptable.

        Args:
            annotation: Single frame annotation from label.json

        Returns:
            True if kinematics are valid
        """
        required_objects = ['Tool 1', 'Tool 2', 'World']

        for obj_name in required_objects:
            obj_data = annotation.get(obj_name, {})

            # Check Position
            position = obj_data.get('Position', [])
            if not position or any(
                p is None or (isinstance(p, float) and math.isnan(p))
                for p in position
            ):
                return False

            # Check Rotation
            rotation = obj_data.get('Rotation', [])
            if not rotation or any(
                r is None or (isinstance(r, float) and math.isnan(r))
                for r in rotation
            ):
                return False

        return True

    def get_sampled_frames(self, trial_id: str) -> List[int]:
        """Get list of sampled frame indices for a trial.

        Samples every 100 frames, skipping those with invalid kinematics
        when kinematics data is available. Falls back to uniform sampling
        by frame count when no kinematics exist (e.g., 6DOF2023).

        Args:
            trial_id: Trial identifier

        Returns:
            List of valid frame indices
        """
        trial = self.trials.get(trial_id)
        if not trial:
            return []

        # Return cached result if available
        if trial.valid_frames is not None:
            return trial.valid_frames

        # Ensure frame count is populated (lazy — first access only)
        self._ensure_frame_count(trial)

        # Load kinematics for NaN checking
        kinematics = self.load_kinematics(trial_id)

        if not kinematics:
            # No kinematics available — sample uniformly by frame count
            valid_frames = list(range(0, trial.total_frames, self.SAMPLE_INTERVAL))
            trial.valid_frames = valid_frames
            return valid_frames

        annotations = kinematics.get('annotations', [])

        # Build frame index lookup
        frame_lookup = {ann.get('Frame', i): ann for i, ann in enumerate(annotations)}

        valid_frames = []
        target_frame = 0

        while target_frame < trial.total_frames:
            # Find next valid frame starting from target
            valid_frame = self._find_next_valid_frame(
                target_frame, trial.total_frames, frame_lookup
            )

            if valid_frame is not None:
                valid_frames.append(valid_frame)
                # Next target is 100 frames after the valid frame we found
                target_frame = valid_frame + self.SAMPLE_INTERVAL
            else:
                # No more valid frames
                break

        trial.valid_frames = valid_frames
        return valid_frames

    def _find_next_valid_frame(
        self,
        start_frame: int,
        max_frame: int,
        frame_lookup: Dict[int, Dict]
    ) -> Optional[int]:
        """Find next valid frame starting from start_frame.

        Args:
            start_frame: Frame to start searching from
            max_frame: Maximum frame number
            frame_lookup: Dict mapping frame index to annotation

        Returns:
            Valid frame index or None if not found
        """
        for frame_idx in range(start_frame, max_frame):
            annotation = frame_lookup.get(frame_idx)
            if annotation and self.has_valid_kinematics(annotation):
                return frame_idx
        return None

    def get_frame_path(self, trial_id: str, frame_idx: int) -> Optional[Path]:
        """Get the path to a specific frame image.

        Args:
            trial_id: Trial identifier
            frame_idx: Frame index

        Returns:
            Path to frame image or None
        """
        trial = self.trials.get(trial_id)
        if not trial:
            return None

        frame_name = trial.frame_pattern.format(frame_idx)
        frame_path = trial.frames_dir / frame_name

        if frame_path.exists():
            return frame_path

        # Try alternative patterns for 6DOF
        if trial.dataset == '6DOF2023':
            # Try simple number format
            alt_path = trial.frames_dir / f"{frame_idx}.bmp"
            if alt_path.exists():
                return alt_path

        return None

    def get_sam_path(self, trial_id: str, frame_idx: int) -> Optional[Path]:
        """Get path to pre-computed SAM masks for a frame.

        Args:
            trial_id: Trial identifier (e.g., "7DOF2024/Trial1")
            frame_idx: Frame index

        Returns:
            Path to .npz file if it exists, else None
        """
        trial = self.trials.get(trial_id)
        if not trial:
            return None

        npz_path = trial.trial_path / "SAM" / f"frame_{frame_idx:05d}.npz"
        if npz_path.exists():
            return npz_path
        return None

    def get_trial_info(self, trial_id: str) -> Optional[TrialInfo]:
        """Get trial information.

        Args:
            trial_id: Trial identifier

        Returns:
            TrialInfo or None
        """
        return self.trials.get(trial_id)

    def move_to_broken(self, trial_id: str, frame_idx: int) -> bool:
        """Move a frame image to the broken/ subdirectory.

        Args:
            trial_id: Trial identifier
            frame_idx: Frame index

        Returns:
            True if moved successfully
        """
        trial = self.trials.get(trial_id)
        if not trial:
            return False

        frame_path = self.get_frame_path(trial_id, frame_idx)
        if not frame_path or not frame_path.exists():
            return False

        # For 6DOF, frames_dir == trial_path (images live directly in trial dir)
        # For 7DOF/BAPES, frames_dir is the Frames/ subdirectory
        broken_dir = trial.frames_dir / 'broken'
        broken_dir.mkdir(exist_ok=True)

        dst = broken_dir / frame_path.name
        try:
            shutil.move(str(frame_path), str(dst))
            return True
        except Exception as e:
            print(f"Error moving {frame_path} to broken/: {e}")
            return False

    def get_all_frame_indices(self, trial_id: str) -> List[int]:
        """Get all frame indices by enumerating actual files on disk.

        Returns sorted list of frame indices that have corresponding image
        files in the trial's frames directory (excludes broken/ subfolder).

        Args:
            trial_id: Trial identifier

        Returns:
            Sorted list of frame indices present on disk.
        """
        trial = self.trials.get(trial_id)
        if not trial:
            return []

        indices: List[int] = []

        if trial.dataset == '6DOF2023':
            if trial.frame_pattern.endswith('.png'):
                # PNG files like test1_0000.png
                for f in trial.frames_dir.glob('*.png'):
                    m = re.search(r'_(\d+)\.png$', f.name)
                    if m:
                        indices.append(int(m.group(1)))
            else:
                # BMP fallback like 0.bmp, 1.bmp
                for f in trial.frames_dir.glob('*.bmp'):
                    if f.parent.name == 'broken':
                        continue
                    m = re.match(r'^(\d+)\.bmp$', f.name)
                    if m:
                        indices.append(int(m.group(1)))
        else:
            # 7DOF2024 / BAPES2024: frame_00050.bmp
            for f in trial.frames_dir.glob('frame_*.bmp'):
                if f.parent.name == 'broken':
                    continue
                m = re.search(r'frame_(\d+)\.bmp$', f.name)
                if m:
                    indices.append(int(m.group(1)))

        indices.sort()
        return indices

    def get_next_frame(self, trial_id: str, current_frame: int) -> Optional[int]:
        """Get next sampled frame after current frame.

        Args:
            trial_id: Trial identifier
            current_frame: Current frame index

        Returns:
            Next frame index or None if at end
        """
        trial = self.trials.get(trial_id)
        if not trial or not trial.valid_frames:
            return None

        for frame in trial.valid_frames:
            if frame > current_frame:
                return frame
        return None

    def get_prev_frame(self, trial_id: str, current_frame: int) -> Optional[int]:
        """Get previous sampled frame before current frame.

        Args:
            trial_id: Trial identifier
            current_frame: Current frame index

        Returns:
            Previous frame index or None if at start
        """
        trial = self.trials.get(trial_id)
        if not trial or not trial.valid_frames:
            return None

        prev_frame = None
        for frame in trial.valid_frames:
            if frame >= current_frame:
                break
            prev_frame = frame
        return prev_frame


# Global frame manager instance
_frame_manager: Optional[FrameManager] = None


def get_frame_manager() -> FrameManager:
    """Get or create the global frame manager instance."""
    global _frame_manager
    if _frame_manager is None:
        _frame_manager = FrameManager()
        _frame_manager.discover_all_trials()
    return _frame_manager
