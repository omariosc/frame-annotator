"""Bulk-mark black-bar artifact frames as broken.

Reads ``outputs/black_frame_scan.json`` (produced by ``src/scan_black_frames.py``)
and sets ``broken=True`` on every affected frame. Frames previously marked
``exclude=True`` have that flag cleared (broken supersedes exclude).

Images are physically moved to a ``broken/`` subfolder via
:meth:`FrameManager.move_to_broken`.

A manifest of all broken frames is written to
``outputs/broken_frames_manifest.json``.

Usage::

    python -m surgical_annotator.mark_broken_frames --dry-run   # preview
    python -m surgical_annotator.mark_broken_frames              # execute

Author: AI-ELT Project
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from surgical_annotator.annotation_store import AnnotationStore
from surgical_annotator.frame_manager import FrameManager

SCAN_PATH = Path("outputs/black_frame_scan.json")
MANIFEST_PATH = Path("outputs/broken_frames_manifest.json")


def mark_broken_frames(
    scan_path: Path = SCAN_PATH,
    manifest_path: Path = MANIFEST_PATH,
    dry_run: bool = False,
) -> dict[str, int]:
    """Mark all black-bar frames as broken in the annotation store.

    Args:
        scan_path: Path to the black frame scan JSON.
        manifest_path: Path to write the broken frames manifest.
        dry_run: If True, report what would be done without writing files.

    Returns:
        Dict with counts: {marked, already_broken, cleared_exclude, total_frames}.
    """
    with open(scan_path, encoding="utf-8") as f:
        scan = json.load(f)

    affected_trials = scan.get("affected_trials", {})
    store = AnnotationStore()
    fm = FrameManager()
    fm.discover_all_trials()

    marked = 0
    already_broken = 0
    cleared_exclude = 0
    moved = 0
    move_failed = 0
    total_frames = 0

    # For manifest
    manifest_trials: dict[str, list[int]] = {}

    for trial_name, trial_info in sorted(affected_trials.items()):
        # Map scan name "Test 10" → annotation trial_id "6DOF2023/Test 10 png"
        trial_id = f"6DOF2023/{trial_name} png"
        frames_dict = trial_info.get("frames", {})
        frame_count = len(frames_dict)
        total_frames += frame_count

        # Load existing annotations
        store.load_trial(trial_id)

        trial_frame_list: list[int] = []

        for frame_idx_str in sorted(frames_dict.keys(), key=int):
            frame_idx = int(frame_idx_str)
            trial_frame_list.append(frame_idx)

            # Check current state
            existing = store.get_frame(trial_id, frame_idx)
            if existing.broken:
                already_broken += 1
                continue

            if dry_run:
                was_excluded = existing.exclude
                print(
                    f"  [DRY RUN] {trial_id} frame {frame_idx}"
                    f"{' (would clear exclude)' if was_excluded else ''}"
                )
                marked += 1
                if was_excluded:
                    cleared_exclude += 1
                continue

            # Move image to broken/ subfolder
            if fm.move_to_broken(trial_id, frame_idx):
                moved += 1
            else:
                move_failed += 1

            # Build update dict
            updates: dict[str, object] = {"broken": True}
            if existing.exclude:
                updates["exclude"] = False
                cleared_exclude += 1

            store.update_frame(trial_id, frame_idx, updates, auto_save=False)
            marked += 1

        # Save all at once per trial (writes both monolithic + per-frame)
        if not dry_run:
            store.save_trial(trial_id)

        manifest_trials[trial_id] = trial_frame_list
        print(f"  {trial_id}: {frame_count} frames processed")

    print()
    print(f"Total black-bar frames:   {total_frames}")
    print(f"Newly marked broken:      {marked}")
    print(f"Already broken:           {already_broken}")
    print(f"Cleared exclude flag:     {cleared_exclude}")
    if not dry_run:
        print(f"Images moved to broken/:  {moved}")
        print(f"Image move failures:      {move_failed}")

    # Write manifest
    if not dry_run:
        manifest = {
            "metadata": {
                "source": str(scan_path),
                "date": datetime.now().isoformat(),
                "total_broken": total_frames,
                "newly_marked": marked,
                "already_broken": already_broken,
                "cleared_exclude": cleared_exclude,
            },
            "trials": {
                tid: sorted(frames)
                for tid, frames in sorted(manifest_trials.items())
            },
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest written to {manifest_path}")

    return {
        "marked": marked,
        "already_broken": already_broken,
        "cleared_exclude": cleared_exclude,
        "total_frames": total_frames,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bulk-mark black-bar artifact frames as broken.",
    )
    parser.add_argument(
        "--scan-path",
        type=Path,
        default=SCAN_PATH,
        help=f"Path to black frame scan JSON (default: {SCAN_PATH}).",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=MANIFEST_PATH,
        help=f"Path for output manifest (default: {MANIFEST_PATH}).",
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

    print(f"Reading black frame scan from {args.scan_path}")
    if args.dry_run:
        print("[DRY RUN MODE - no changes will be made]\n")
    else:
        print()

    mark_broken_frames(args.scan_path, args.manifest_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
