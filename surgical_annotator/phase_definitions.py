"""Phase and object annotation taxonomy for peg transfer tasks.

Defines the hierarchical annotation schema:
- Coarse phases (7): idle, reach, nudge, grasp, transfer, place, dropped
- Fine sub-phases (20): granular actions within each coarse phase
- Atomic events (12): instantaneous events that can co-occur with any phase
- Peg states (7): state machine for each peg's lifecycle
- Colors for visualization

Based on JIGSAWS, PETRAW, LASANA, and FLS literature.
"""

from __future__ import annotations

# ============================================================================
# Phase Taxonomy (3-level hierarchy)
# ============================================================================

COARSE_PHASES: list[str] = [
    'idle',
    'reach',
    'nudge',
    'grasp',
    'transfer',
    'place',
    'dropped',
]

FINE_PHASES: dict[str, list[str]] = {
    'idle': ['waiting', 'planning', 'repositioning'],
    'reach': ['approach_peg', 'open_jaw_prepare', 'approach_target_post'],
    'nudge': ['align_peg', 'push_peg', 'reposition_peg'],
    'grasp': ['position_jaw', 'close_jaw', 'lift_peg', 'verify_grasp'],
    'transfer': ['approach_partner', 'align_tools', 'handoff', 'verify_transfer'],
    'place': ['approach_post', 'align_peg', 'release_peg', 'verify_place'],
    'dropped': ['peg_dropped'],
}

ATOMIC_EVENTS: list[str] = [
    'jaw_open',
    'jaw_close',
    'peg_contact',
    'peg_release',
    'tool_contact',
    'fumble',
    'peg_drop',
    'correction',
    'hesitation',
    'regrasp',
    'wrong_placement',
    'collision',
]

# ============================================================================
# Peg & Object Taxonomy
# ============================================================================

PEG_STATES: list[str] = [
    'on_source_post',
    'grasped_by_tool1',
    'grasped_by_tool2',
    'in_transfer',
    'on_target_post',
    'dropped',
    'out_of_view',
]

NUM_PEGS: int = 6
NUM_SOURCE_POSTS: int = 6
NUM_TARGET_POSTS: int = 6

# ============================================================================
# Visualization Colors
# ============================================================================

PHASE_COLORS: dict[str, dict[str, str]] = {
    'idle':     {'bg': '#6b7280', 'text': '#fff'},
    'reach':    {'bg': '#3b82f6', 'text': '#fff'},
    'nudge':    {'bg': '#f97316', 'text': '#fff'},
    'grasp':    {'bg': '#f59e0b', 'text': '#000'},
    'transfer': {'bg': '#8b5cf6', 'text': '#fff'},
    'place':    {'bg': '#10b981', 'text': '#fff'},
    'dropped':  {'bg': '#ef4444', 'text': '#fff'},
}

PEG_COLORS: list[str] = [
    '#f97316',  # Peg 1: Orange
    '#a855f7',  # Peg 2: Purple
    '#06b6d4',  # Peg 3: Cyan
    '#eab308',  # Peg 4: Yellow
    '#ec4899',  # Peg 5: Pink
    '#84cc16',  # Peg 6: Lime
]

# ============================================================================
# Keyboard / Mouse Mapping Guide
# ============================================================================

PHASE_KEYBINDS: dict[str, str] = {
    'Z': 'idle',
    'X': 'reach',
    'C': 'nudge',
    'D': 'grasp',
    'R': 'transfer',
    'T': 'place',
    'G': 'dropped',
}

MOUSE_MAPPING_GUIDE: list[dict[str, str]] = [
    {'button': 'Side Button 1 (easy)',  'key': 'X', 'phase': 'Reach',    'freq': '~35%'},
    {'button': 'Side Button 2 (easy)',  'key': 'G', 'phase': 'Dropped',  'freq': 'Rare'},
    {'button': 'Button 3 (medium)',     'key': 'C', 'phase': 'Nudge',    'freq': '~5%'},
    {'button': 'Button 4 (medium)',     'key': 'D', 'phase': 'Grasp',    'freq': '~12%'},
    {'button': 'Button 5 (medium)',     'key': 'R', 'phase': 'Transfer', 'freq': '~15%'},
    {'button': 'Button 6 (medium)',     'key': 'T', 'phase': 'Place',    'freq': '~13%'},
    {'button': 'Button 7 (harder)',     'key': 'Z', 'phase': 'Idle',     'freq': 'Rare'},
    {'button': 'Button 8 (harder)',     'key': 'H', 'phase': 'Next Cycle', 'freq': 'Per cycle'},
]


def get_definitions_dict() -> dict:
    """Return all definitions as a JSON-serializable dict for the frontend."""
    return {
        'coarse_phases': COARSE_PHASES,
        'fine_phases': FINE_PHASES,
        'atomic_events': ATOMIC_EVENTS,
        'peg_states': PEG_STATES,
        'num_pegs': NUM_PEGS,
        'phase_colors': PHASE_COLORS,
        'peg_colors': PEG_COLORS,
        'phase_keybinds': PHASE_KEYBINDS,
        'mouse_mapping': MOUSE_MAPPING_GUIDE,
    }
