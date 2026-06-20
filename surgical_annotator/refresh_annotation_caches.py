#!/usr/bin/env python
"""One-time refresh of all annotation caches.

Restores any missing per-frame JSONs from backups and monolithic files,
then regenerates every ``_progress.json`` cache from actual frame data.

Usage::

    python scripts/refresh_annotation_caches.py [--dry-run]
"""

import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# Force UTF-8 output on Windows to avoid charmap encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from surgical_annotator.annotation_store import FrameAnnotation, ANNOTATION_DIR


def _parse_backup_timestamp(stem: str) -> tuple[str, str] | None:
    """Extract (trial_name, timestamp) from a backup filename stem.

    Backup filenames follow the pattern ``{trial_name}_{YYYYMMDD_HHMMSS}``.

    Args:
        stem: Filename without extension.

    Returns:
        (trial_name, timestamp_str) or None if the pattern doesn't match.
    """
    m = re.match(r"^(.+)_(\d{8}_\d{6})$", stem)
    if m:
        return m.group(1), m.group(2)
    return None


def _load_frames_from_json(path: Path) -> dict[str, dict]:
    """Load the ``frames`` dict from a monolithic or backup JSON.

    Args:
        path: Path to the JSON file.

    Returns:
        Dict mapping frame-index strings to frame data dicts.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("frames", {})


def _frame_path(trial_dir: Path, frame_idx: int) -> Path:
    """Per-frame file path matching the annotation store convention."""
    return trial_dir / f"frame_{frame_idx:04d}.json"


def _write_frame_atomic(path: Path, data: dict) -> None:
    """Write a per-frame JSON atomically via temp file + rename.

    Args:
        path: Target file path.
        data: Frame annotation dict.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        # On Windows, target must not exist for os.rename; use os.replace.
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _classify_frame(data: dict) -> str:
    """Classify a frame dict into a status category.

    Uses ``FrameAnnotation.from_dict()`` for consistent logic.

    Args:
        data: Raw frame dict from JSON.

    Returns:
        One of: ``broken``, ``skipped``, ``negative``, ``completed``, ``partial``.
    """
    ann = FrameAnnotation.from_dict(data)
    if ann.broken:
        return "broken"
    if ann.skipped:
        return "skipped"
    if ann.is_negative():
        return "negative"
    if ann.is_complete():
        return "completed"
    return "partial"


# ── Step 1: Restore from backups ─────────────────────────────────────


def restore_from_backups(dry_run: bool = False) -> int:
    """Restore missing per-frame files from the latest backup per trial.

    Args:
        dry_run: If True, report what would be restored without writing.

    Returns:
        Number of frames restored.
    """
    backup_dir = ANNOTATION_DIR / "backups"
    if not backup_dir.is_dir():
        print("  No backups/ directory found — skipping.")
        return 0

    # Group backup files by trial name, pick latest timestamp
    latest: dict[str, Path] = {}
    for bp in backup_dir.glob("*.json"):
        parsed = _parse_backup_timestamp(bp.stem)
        if not parsed:
            continue
        trial_name, ts = parsed
        if trial_name not in latest or ts > _parse_backup_timestamp(latest[trial_name].stem)[1]:
            latest[trial_name] = bp

    print(f"  Found latest backups for {len(latest)} trials.")

    restored = 0
    for trial_name, bp in sorted(latest.items()):
        trial_dir = ANNOTATION_DIR / trial_name
        frames = _load_frames_from_json(bp)
        trial_restored = 0
        for frame_str, frame_data in frames.items():
            frame_idx = int(frame_str)
            fp = _frame_path(trial_dir, frame_idx)
            if not fp.exists():
                if dry_run:
                    print(f"    [DRY RUN] Would restore {fp.name} in {trial_name}")
                else:
                    # Ensure frame_idx is set in the data
                    frame_data["frame_idx"] = frame_idx
                    _write_frame_atomic(fp, frame_data)
                trial_restored += 1
        if trial_restored:
            print(f"    {trial_name}: restored {trial_restored} frames from backup")
        restored += trial_restored

    return restored


# ── Step 2: Restore from monolithic JSONs ────────────────────────────


def restore_from_monolithic(dry_run: bool = False) -> int:
    """Restore missing per-frame files from top-level monolithic JSONs.

    Args:
        dry_run: If True, report what would be restored without writing.

    Returns:
        Number of frames restored.
    """
    monolithic_files = sorted(ANNOTATION_DIR.glob("*.json"))
    print(f"  Found {len(monolithic_files)} monolithic JSON files.")

    restored = 0
    for mf in monolithic_files:
        trial_name = mf.stem  # e.g. "6DOF2023_Test 1 png"
        trial_dir = ANNOTATION_DIR / trial_name
        frames = _load_frames_from_json(mf)
        if not frames:
            continue
        trial_restored = 0
        for frame_str, frame_data in frames.items():
            frame_idx = int(frame_str)
            fp = _frame_path(trial_dir, frame_idx)
            if not fp.exists():
                if dry_run:
                    print(f"    [DRY RUN] Would restore {fp.name} in {trial_name}")
                else:
                    frame_data["frame_idx"] = frame_idx
                    _write_frame_atomic(fp, frame_data)
                trial_restored += 1
        if trial_restored:
            print(f"    {trial_name}: restored {trial_restored} frames from monolithic")
        restored += trial_restored

    return restored


# ── Step 3: Regenerate all _progress.json caches ────────────────────


def regenerate_progress_caches(dry_run: bool = False) -> tuple[int, dict[str, int]]:
    """Regenerate ``_progress.json`` for every trial directory.

    Args:
        dry_run: If True, compute but don't write.

    Returns:
        (number of caches written, overall status totals).
    """
    totals: dict[str, int] = defaultdict(int)
    written = 0
    errors = 0

    trial_dirs = sorted(
        d for d in ANNOTATION_DIR.iterdir()
        if d.is_dir() and d.name != "backups"
    )

    n_trials = len(trial_dirs)
    for i, trial_dir in enumerate(trial_dirs, 1):
        frame_files = list(trial_dir.glob("frame_*.json"))
        if not frame_files:
            continue

        counts: dict[str, int] = {
            "completed": 0,
            "skipped": 0,
            "negative": 0,
            "partial": 0,
            "broken": 0,
        }

        for ff in frame_files:
            try:
                with open(ff, "r") as f:
                    data = json.load(f)
                status = _classify_frame(data)
                counts[status] += 1
                totals[status] += 1
            except Exception as e:
                print(f"    ERROR reading {ff}: {e}", flush=True)
                errors += 1

        total_frames = sum(counts.values())
        summary_line = ", ".join(f"{k}={v}" for k, v in counts.items() if v)

        if dry_run:
            print(f"  [{i}/{n_trials}] {trial_dir.name}: {total_frames} frames -- {summary_line}", flush=True)
        else:
            summary_path = trial_dir / "_progress.json"
            try:
                with open(summary_path, "w") as f:
                    json.dump(counts, f)
            except Exception as e:
                print(f"    ERROR writing {summary_path}: {e}", flush=True)
                errors += 1
                continue
            # Print progress every 10 trials or on the last one
            if i % 10 == 0 or i == n_trials:
                print(f"  [{i}/{n_trials}] processed...", flush=True)

        written += 1

    totals["_errors"] = errors
    return written, dict(totals)


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore missing annotations and regenerate progress caches."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without writing any files.",
    )
    args = parser.parse_args()

    print(f"Annotation directory: {ANNOTATION_DIR}", flush=True)
    if args.dry_run:
        print("*** DRY RUN -- no files will be written ***\n", flush=True)

    # Step 1
    print("Step 1: Restoring missing frames from backups...", flush=True)
    backup_restored = restore_from_backups(dry_run=args.dry_run)
    print(f"  => Restored {backup_restored} frames from backups.\n", flush=True)

    # Step 2
    print("Step 2: Restoring missing frames from monolithic JSONs...", flush=True)
    mono_restored = restore_from_monolithic(dry_run=args.dry_run)
    print(f"  => Restored {mono_restored} frames from monolithic files.\n", flush=True)

    # Step 3
    print("Step 3: Regenerating _progress.json caches...", flush=True)
    caches_written, totals = regenerate_progress_caches(dry_run=args.dry_run)
    errors = totals.pop("_errors", 0)
    total_frames = sum(totals.values())
    print(f"  => Regenerated {caches_written} _progress.json caches.\n", flush=True)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Frames restored from backups:    {backup_restored}")
    print(f"  Frames restored from monolithic: {mono_restored}")
    print(f"  Progress caches regenerated:     {caches_written}")
    print(f"  Total frames across all trials:  {total_frames}")
    if totals:
        for status, count in sorted(totals.items()):
            print(f"    {status:>12}: {count}")
    if errors:
        print(f"  Errors encountered: {errors}")
    print("=" * 60)


if __name__ == "__main__":
    main()
