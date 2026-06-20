"""Migrate single-tool `coarse` phase to per-tool tool1/tool2 schema.

Rules (peg-transfer 6DOF Test 1):
- Cycles 0-5: primary tool = Tool 1, secondary tool = Tool 2
- Cycles 6-11: primary tool = Tool 2, secondary tool = Tool 1
- 'transfer' phase → both tools annotated as 'transfer'
- 'place' phase → only the secondary tool (other tool is idle)
- Any other phase → only the primary tool (other tool is idle)
- 'idle' phase → both tools idle

Run:
    python -m surgical_annotator.migrate_phases_to_dual_tool [trial_dir ...]

Default targets (run with no args):
    outputs/annotations/6DOF2023_Test 1 png
    outputs/annotations/6DOF2023_Test 10 png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


DEFAULT_TARGETS = [
    'outputs/annotations/6DOF2023_Test 1 png',
    'outputs/annotations/6DOF2023_Test 10 png',
]


def derive_dual_tool(coarse: str, cycle_index: int) -> tuple[str, str]:
    """Return (tool1, tool2) phases per the peg-transfer rule."""
    co = (coarse or '').lower()
    if not co:
        return 'idle', 'idle'
    primary_is_tool1 = (cycle_index <= 5)
    if co == 'idle':
        return 'idle', 'idle'
    if co == 'transfer':
        return 'transfer', 'transfer'
    if co == 'place':
        # place is on the secondary tool
        return ('idle', 'place') if primary_is_tool1 else ('place', 'idle')
    # All other phases (reach, nudge, grasp, return, ...) on primary
    return (co, 'idle') if primary_is_tool1 else ('idle', co)


def migrate_frame_file(path: Path, fill_idle: bool = False, inherit_cycle: int = 0) -> bool:
    """Migrate a single per-frame JSON file. Returns True if changed.

    Args:
        path: per-frame JSON path
        fill_idle: when True, frames with empty `coarse` are treated as idle
                   (so timeline gaps in fully-annotated trials display as
                    idle/idle instead of being excluded). Off by default to
                   preserve unannotated frames in partially-annotated trials.
    """
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"  ! skip {path.name}: {e}")
        return False

    phase = data.get('phase')
    if not isinstance(phase, dict):
        if not fill_idle:
            return False
        # No phase field at all — create a fresh idle one so the timeline has
        # no gaps. Cycle is unknown; default to 0 (will be picked up by user).
        phase = {}
        data['phase'] = phase

    original_coarse = phase.get('coarse', '') or ''
    cycle_index = phase.get('cycle_index', 0) or 0
    # If this frame had no phase at all, inherit cycle from preceding annotated frame.
    if 'cycle_index' not in phase:
        cycle_index = inherit_cycle
        phase['cycle_index'] = cycle_index
    fill_now = (not original_coarse and fill_idle)
    effective_coarse = 'idle' if fill_now else original_coarse

    t1, t2 = derive_dual_tool(effective_coarse, cycle_index)

    # Skip only if everything is already exactly right.
    if (
        not fill_now
        and phase.get('tool1') == t1
        and phase.get('tool2') == t2
        and 'tool1' in phase
    ):
        return False

    if fill_now:
        phase['coarse'] = 'idle'
    phase['tool1'] = t1
    phase['tool2'] = t2
    # active_tool: prefer non-idle tool's number; default to 1
    if t1 != 'idle' and t2 == 'idle':
        phase['active_tool'] = 1
    elif t2 != 'idle' and t1 == 'idle':
        phase['active_tool'] = 2
    elif phase.get('active_tool') not in (1, 2):
        phase['active_tool'] = 1 if cycle_index <= 5 else 2

    # Preserve legacy `coarse` for back-compat readers
    phase['coarse'] = effective_coarse

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    return True


def migrate_monolithic_file(path: Path) -> int:
    """Migrate a legacy monolithic *.json file. Returns count modified."""
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"  ! skip {path.name}: {e}")
        return 0
    frames = data.get('frames', {})
    if not isinstance(frames, dict):
        return 0
    n = 0
    for _, fr in frames.items():
        phase = fr.get('phase')
        if not isinstance(phase, dict):
            continue
        coarse = phase.get('coarse', '') or ''
        cycle_index = phase.get('cycle_index', 0) or 0
        t1, t2 = derive_dual_tool(coarse, cycle_index)
        if phase.get('tool1') == t1 and phase.get('tool2') == t2 and 'tool1' in phase:
            continue
        phase['tool1'] = t1
        phase['tool2'] = t2
        if t1 != 'idle' and t2 == 'idle':
            phase['active_tool'] = 1
        elif t2 != 'idle' and t1 == 'idle':
            phase['active_tool'] = 2
        elif phase.get('active_tool') not in (1, 2):
            phase['active_tool'] = 1 if cycle_index <= 5 else 2
        n += 1
    if n:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    return n


def migrate_target(root: Path, target: str, fill_idle: bool = False) -> None:
    target_dir = root / target
    target_json = root / (target + '.json')

    print(f"=== {target} (fill_idle={fill_idle}) ===")
    if target_dir.is_dir():
        # First pass: collect cycle_index per frame so gaps inherit neighbour's cycle.
        files = sorted(target_dir.glob('frame_*.json'))
        cycle_by_idx: dict[int, int] = {}
        for p in files:
            try:
                data = json.loads(p.read_text(encoding='utf-8'))
                ph = data.get('phase')
                if isinstance(ph, dict) and ph.get('coarse'):
                    idx = int(p.stem.split('_')[1])
                    cycle_by_idx[idx] = ph.get('cycle_index', 0) or 0
            except Exception:
                pass

        # Build inherited cycle: forward-fill from preceding annotated frame.
        inherited: dict[int, int] = {}
        last = 0
        for p in files:
            try:
                idx = int(p.stem.split('_')[1])
            except Exception:
                continue
            if idx in cycle_by_idx:
                last = cycle_by_idx[idx]
            inherited[idx] = last

        changed = 0
        total = 0
        for p in files:
            total += 1
            try:
                idx = int(p.stem.split('_')[1])
            except Exception:
                idx = None
            inherit_cycle = inherited.get(idx, 0) if idx is not None else 0
            if migrate_frame_file(p, fill_idle=fill_idle, inherit_cycle=inherit_cycle):
                changed += 1
        print(f"  per-frame: {changed}/{total} files updated")
    else:
        print("  (no per-frame directory)")

    if target_json.is_file():
        n = migrate_monolithic_file(target_json)
        print(f"  monolithic: {n} frame entries updated")
    else:
        print("  (no monolithic .json)")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    args = sys.argv[1:]
    fill_idle = False
    if '--fill-idle' in args:
        fill_idle = True
        args = [a for a in args if a != '--fill-idle']
    targets = args or DEFAULT_TARGETS
    for t in targets:
        # Trial 1 is fully annotated → fill empty-coarse gaps as idle.
        # Trial 10 is partial → leave unannotated frames blank.
        per_target_fill = fill_idle or ('Test 1 png' in t)
        migrate_target(root, t, fill_idle=per_target_fill)
    print("done.")


if __name__ == '__main__':
    main()
