"""Migrate annotations: remove all keypoints and simplify visibility.

This script migrates existing annotation files to the simplified format:
- Removes all keypoint data (tool*_joint, tool*_ee_tip, tool*_ee_left, tool*_ee_right)
- Removes all keypoint-related line data (middle, joint_ee_tip, ee_left, ee_right from lines)
- Simplifies visibility to just {mask: 0|1, lines: 0|1}
- Preserves mask and top/bottom shaft lines

Visibility: 1=visible (requires annotation), 0=occluded (no annotation needed).
Old -1 (out of scene) is treated as 0 (occluded).

Run: python -m surgical_annotator.migrate_annotations
"""

import json
from pathlib import Path


def migrate_file(filepath: Path) -> bool:
    """Migrate a single annotation file.

    Args:
        filepath: Path to the annotation JSON file

    Returns:
        True if file was modified, False otherwise
    """
    with open(filepath) as f:
        data = json.load(f)

    modified = False

    for frame_id, frame in data.get('frames', {}).items():
        # Remove keypoint fields
        for key in ['tool1_joint', 'tool2_joint',
                    'tool1_ee_tip', 'tool2_ee_tip',
                    'tool1_ee_left', 'tool1_ee_right',
                    'tool2_ee_left', 'tool2_ee_right',
                    'tool1_tooltip', 'tool2_tooltip']:  # Also old tooltip keys
            if key in frame:
                del frame[key]
                modified = True

        # Simplify lines: keep only top/bottom
        for lines_key in ['tool1_lines', 'tool2_lines']:
            if lines_key in frame and isinstance(frame[lines_key], dict):
                old_lines = frame[lines_key]
                new_lines = {
                    'top': old_lines.get('top', []),
                    'bottom': old_lines.get('bottom', [])
                }
                # Check if we're removing any extra keys
                extra_keys = set(old_lines.keys()) - {'top', 'bottom'}
                if extra_keys:
                    frame[lines_key] = new_lines
                    modified = True

        # Simplify visibility: keep only mask/lines, normalize -1 to 0
        for vis_key in ['tool1_visibility', 'tool2_visibility']:
            if vis_key in frame and isinstance(frame[vis_key], dict):
                old_vis = frame[vis_key]
                # Get mask and lines, treating -1 as 0
                mask_val = old_vis.get('mask', 1)
                lines_val = old_vis.get('lines', 1)
                new_vis = {
                    'mask': 1 if mask_val == 1 else 0,
                    'lines': 1 if lines_val == 1 else 0
                }
                # Check if we're changing anything
                if old_vis != new_vis:
                    frame[vis_key] = new_vis
                    modified = True

        # Remove old missing keys
        for missing_key in ['tool1_missing', 'tool2_missing']:
            if missing_key in frame:
                del frame[missing_key]
                modified = True

    if modified:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    return modified


def main():
    """Run migration on all annotation files."""
    annotations_dir = Path(__file__).parent.parent.parent / 'outputs' / 'annotations'

    if not annotations_dir.exists():
        print(f"Annotations directory not found: {annotations_dir}")
        return

    migrated_count = 0
    unchanged_count = 0

    for filepath in sorted(annotations_dir.glob('*.json')):
        try:
            if migrate_file(filepath):
                print(f"Migrated: {filepath.name}")
                migrated_count += 1
            else:
                print(f"No changes: {filepath.name}")
                unchanged_count += 1
        except Exception as e:
            print(f"Error migrating {filepath.name}: {e}")

    print(f"\nMigration complete: {migrated_count} files modified, {unchanged_count} unchanged")


if __name__ == '__main__':
    main()
