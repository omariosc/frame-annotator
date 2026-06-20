"""Enforce monotonic non-decreasing `cycle_index` within a trial.

Walks per-frame JSONs in frame-index order. If a frame's cycle_index drops
below the running maximum, snap it up to the running max. After the cycle
fix, re-derive tool1/tool2 using the same primary-tool rule as the dual-tool
migration (C0-C5 primary=Tool 1, C6+ primary=Tool 2; transfer→both,
place→secondary tool only).

Usage::

    python -m surgical_annotator.enforce_monotonic_cycle
    python -m surgical_annotator.enforce_monotonic_cycle "outputs/annotations/<trial>"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from surgical_annotator.migrate_phases_to_dual_tool import derive_dual_tool


DEFAULT_TARGETS = [
    'outputs/annotations/6DOF2023_Test 1 png',
    'outputs/annotations/6DOF2023_Test 10 png',
]


def fix_trial(target_dir: Path) -> tuple[int, int]:
    """Returns (frames_changed, drops_fixed)."""
    files = sorted(
        target_dir.glob('frame_*.json'),
        key=lambda p: int(p.stem.split('_')[1]),
    )
    running_max = 0
    frames_changed = 0
    drops_fixed = 0

    for f in files:
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
        except Exception:
            continue
        phase = data.get('phase')
        if not isinstance(phase, dict):
            continue
        ci = phase.get('cycle_index', 0) or 0
        coarse = phase.get('coarse', '') or ''

        new_ci = ci
        if ci < running_max:
            new_ci = running_max
            drops_fixed += 1
        else:
            running_max = ci

        # Re-derive tool1/tool2 from (coarse, new_ci) under the rule.
        if coarse:
            t1, t2 = derive_dual_tool(coarse, new_ci)
        else:
            t1, t2 = phase.get('tool1', 'idle'), phase.get('tool2', 'idle')

        if (
            phase.get('cycle_index') == new_ci
            and phase.get('tool1') == t1
            and phase.get('tool2') == t2
        ):
            continue

        phase['cycle_index'] = new_ci
        phase['tool1'] = t1
        phase['tool2'] = t2
        # Keep active_tool consistent with the rule when possible.
        if t1 != 'idle' and t2 == 'idle':
            phase['active_tool'] = 1
        elif t2 != 'idle' and t1 == 'idle':
            phase['active_tool'] = 2

        f.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        frames_changed += 1

    return frames_changed, drops_fixed


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    targets = sys.argv[1:] or DEFAULT_TARGETS
    for t in targets:
        target_dir = root / t
        print(f"=== {t} ===")
        if not target_dir.is_dir():
            print("  (not a directory, skipping)")
            continue
        changed, drops = fix_trial(target_dir)
        print(f"  drops fixed: {drops}, frames rewritten: {changed}")


if __name__ == '__main__':
    main()
