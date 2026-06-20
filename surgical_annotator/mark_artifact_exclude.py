"""Bulk-mark RGB artifact frames as exclude=True.

One-off script that reads ``outputs/rgb_artifact_scan.json`` and sets
``exclude=True`` on every affected frame's per-frame annotation file.
Frames marked ``exclude`` are still usable for training but are kept
out of the validation split during YOLO export.

Usage::

    python -m surgical_annotator.mark_artifact_exclude
    python -m surgical_annotator.mark_artifact_exclude --dry-run

Author: AI-ELT Project
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from surgical_annotator.annotation_store import AnnotationStore


SCAN_PATH = Path("outputs/rgb_artifact_scan.json")


def mark_artifact_frames(
    scan_path: Path = SCAN_PATH,
    dry_run: bool = False,
) -> dict[str, int]:
    """Mark all artifact frames as exclude=True in the annotation store.

    Args:
        scan_path: Path to the RGB artifact scan JSON.
        dry_run: If True, report what would be done without writing files.

    Returns:
        Dict with counts: {marked, skipped, total_frames}.
    """
    with open(scan_path, encoding="utf-8") as f:
        scan = json.load(f)

    affected_trials = scan.get("affected_trials", {})
    store = AnnotationStore()

    marked = 0
    already_excluded = 0
    total_frames = 0

    for trial_id, trial_info in affected_trials.items():
        frames_list = trial_info.get("frames", [])
        total_frames += len(frames_list)

        # Load existing annotations for this trial
        store.load_trial(trial_id)

        for frame_info in frames_list:
            # Parse frame index from filename like "frame_00022.bmp"
            frame_name: str = frame_info["frame"]
            frame_idx = int(
                frame_name.replace("frame_", "").replace(".bmp", "")
            )

            # Check current state
            existing = store.get_frame(trial_id, frame_idx)
            if existing.exclude:
                already_excluded += 1
                continue

            if dry_run:
                print(f"  [DRY RUN] Would mark {trial_id} frame {frame_idx}")
                marked += 1
                continue

            store.update_frame(
                trial_id, frame_idx, {"exclude": True}, auto_save=True
            )
            marked += 1

        trial_count = len(frames_list)
        print(f"  {trial_id}: {trial_count} frames processed")

    print()
    print(f"Total artifact frames:  {total_frames}")
    print(f"Newly marked exclude:   {marked}")
    print(f"Already excluded:       {already_excluded}")

    return {
        "marked": marked,
        "already_excluded": already_excluded,
        "total_frames": total_frames,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bulk-mark RGB artifact frames as exclude=True.",
    )
    parser.add_argument(
        "--scan-path",
        type=Path,
        default=SCAN_PATH,
        help=f"Path to artifact scan JSON (default: {SCAN_PATH}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without writing files.",
    )
    args = parser.parse_args()

    if not args.scan_path.exists():
        print(f"Error: scan file not found: {args.scan_path}")
        return

    print(f"Reading artifact scan from {args.scan_path}")
    print()
    mark_artifact_frames(args.scan_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
