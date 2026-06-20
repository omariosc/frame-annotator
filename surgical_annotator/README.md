# Surgical Image Annotation Tool

A web-based annotation tool for labeling surgical tool masks, shaft lines, keypoints, and phase annotations in endoscopic images. Designed for creating training data for computer vision models for surgical video.

---

## Quick Start

### Windows (omarpc — default)

```bash
# From the repo root
python -m surgical_annotator

# Opens at http://localhost:5000
```

No arguments needed — defaults to `./data`.

### macOS / Portable (offline use)

```bash
# 1. Clone the repo
git clone https://github.com/omariosc/frame-annotator.git
cd frame-annotator

# 2. Install minimal dependencies (no GPU/PyTorch needed)
pip install flask numpy Pillow

# 3. Copy your data to a local directory (or use an external drive)
#    The data dir must contain: 6DOF2023/, 7DOF2024/, BAPES2024/
#    Annotations are stored in: <data-dir>/outputs/annotations/

# 4. Run with --data-dir pointing to your data
python -m surgical_annotator --data-dir /path/to/your/data

# Opens at http://localhost:5000
```

### CLI Options

```
python -m surgical_annotator [OPTIONS]

Options:
  --data-dir PATH   Root data directory containing 6DOF2023/, 7DOF2024/,
                    BAPES2024/, outputs/. Defaults to ./data.
  --port PORT       Port to run the server on (default: 5000)
  --host HOST       Host to bind to (default: 0.0.0.0)
```

### Data Directory Structure

Your `--data-dir` should look like this:

```
/path/to/your/data/
├── 6DOF2023/
│   ├── Test 1 png/           # PNG frames (preferred)
│   │   ├── test1_0000.png
│   │   ├── test1_0001.png
│   │   └── ...
│   ├── Test 1/               # BMP frames (fallback)
│   │   ├── 0.bmp
│   │   └── ...
│   └── ...
├── 7DOF2024/
│   ├── Attempt1/
│   │   └── Trial1/
│   │       ├── Frames/
│   │       │   ├── frame_00000.bmp
│   │       │   └── ...
│   │       └── label.json
│   ├── Trial33/              # top-level trials (33-60)
│   │   ├── Frames/
│   │   └── label.json
│   └── ...
├── BAPES2024/
│   ├── Industry/
│   │   └── Trial1/
│   │       ├── Frames/
│   │       └── label.json
│   └── MIS Course/
│       └── ...
└── outputs/
    └── annotations/          # created automatically
        ├── 6DOF2023_Test 1 png/
        │   ├── frame_0000.json
        │   └── ...
        └── 7DOF2024_Trial1/
            └── ...
```

**Tip for flights**: Copy just the datasets you need (e.g., only `6DOF2023/`) to save space. Missing datasets are silently skipped.

### Syncing Annotations Back

After annotating offline, copy the annotations back:

```bash
# From your macOS machine, sync annotations to Windows
rsync -av /path/to/data/outputs/annotations/ user@omarpc:/d/Data/AI-ELT/outputs/annotations/

# Or just copy the folder manually via USB/cloud
```

Alternatively, use the **Copy** button in the annotation panel to copy frame annotations as JSON to clipboard, then **Apply** on the other machine.

---

## Overview

The annotation tool supports labeling **two surgical tools per frame** with **12 annotation components** total:
- **Tool 1** (Blue): Left-hand tool
- **Tool 2** (Green): Right-hand tool

Each tool requires up to **6 components**:
- **Mask polygon**: Outline of the visible tool surface
- **Shaft lines**: Top and bottom edge lines along the tool shaft (midline auto-computed)
- **4 keypoints**: Joint, End-Effector Tip, EE Left, EE Right

---

## UI Layout

```
+---------------------------------------------------------------------+
|                                                                     |
|  +--------------+  +-------------------------+  +----------------+  |
|  | LEFT SIDEBAR |  |      CANVAS AREA        |  | RIGHT SIDEBAR  |  |
|  |              |  |                         |  |                |  |
|  | * Dataset    |  |                         |  | * Tool 1       |  |
|  | * Trial      |  |     [Image + Overlay]   |  |   - Visibility |  |
|  |   (dropdown) |  |                         |  |   - Mask (1)   |  |
|  | * Prev/Next  |  |                         |  |   - Top (2)    |  |
|  |   Trial btns |  |                         |  |   - Bottom (3) |  |
|  | * Progress   |  |                         |  |   - Joint (7)  |  |
|  | * Navigation |  |                         |  |   - EE Tip (8) |  |
|  | * Status     |  |                         |  |   - EE Left (Q)|  |
|  | * Options    |  |                         |  |   - EE Right(W)|  |
|  |   [x] Auto-  |  |                         |  |                |  |
|  |    advance   |  |                         |  | * Tool 2       |  |
|  | * Mouse help |  |                         |  |   - Visibility |  |
|  |              |  |                         |  |   - Mask (4)   |  |
|  | [Batch Mode] |  |  [Zoom] [Tool] [Info]   |  |   - Top (5)    |  |
|  |              |  |                         |  |   - Bottom (6) |  |
|  |              |  |                         |  |   - Joint (9)  |  |
|  |              |  |                         |  |   - EE Tip (0) |  |
|  |              |  |                         |  |   - EE Left (O)|  |
|  |              |  |                         |  |   - EE Right(I)|  |
|  |              |  |                         |  |                |  |
|  |              |  |                         |  | * SAM Panel    |  |
|  |              |  |                         |  | * JSON Viewer  |  |
|  +--------------+  +-------------------------+  +----------------+  |
|                                                                     |
+---------------------------------------------------------------------+
```

### Left Sidebar
- **Dataset & Trial**: Select dataset (6DOF2023, 7DOF2024, BAPES2024) and trial
- **Trial Navigation**: Prev/Next trial buttons for quick switching
- **Progress**: Visual progress bar with completed/skipped/negative counts
- **Navigation**: Frame controls, jump-to-frame, sample rate selector
- **Status Checklist**: Quick view of what's annotated (clickable to select tool)
- **Options**: Toggle pose direction overlay, auto-advance checkbox
- **Batch Mode**: Button to enter batch annotation grid view
- **Mouse Help**: Reference for mouse controls

### Center Canvas
- **Image display**: Current frame with annotation overlay
- **Zoom controls**: +/- buttons, fit-to-view, zoom level indicator
- **Tool indicator**: Shows currently selected annotation tool
- **Kinematics info**: Displays kinematic data when available

### Right Sidebar
- **Tool 1 (Blue)**: Visibility controls + mask/line/keypoint tools
- **Tool 2 (Green)**: Visibility controls + mask/line/keypoint tools
- **SAM Panel**: AI-assisted segmentation controls
- **JSON Viewer**: Live annotation JSON panel with copy-to-clipboard

---

## Keyboard Shortcuts

### Tool Selection
| Key | Tool 1 | Key | Tool 2 |
|-----|--------|-----|--------|
| `1` | Mask | `4` | Mask |
| `2` | Top Line | `5` | Top Line |
| `3` | Bottom Line | `6` | Bottom Line |
| `7` | Joint | `9` | Joint |
| `8` | EE Tip | `0` | EE Tip |
| `Q` | EE Left | `O` | EE Left |
| `W` | EE Right | `I` | EE Right |

### Navigation
| Key | Action |
|-----|--------|
| `<-` | Previous frame |
| `->` | Next frame |
| `S` | Skip / Unskip frame |

### Editing
| Key | Action |
|-----|--------|
| `E` | Toggle edit mode (drag vertices) |
| `P` | Copy prior frame annotations |
| `C` | Clear all annotations (with confirmation) |
| `Backspace` | Clear current tool |
| `Delete` | Delete selected vertices/keypoint (in edit mode) |
| `Ctrl+Z` | Undo last action |
| `Ctrl+S` | Force save |

### Visibility / Occlusion
| Key | Action |
|-----|--------|
| `N` | Mark current tool occluded + advance to next tool |
| `Tab` | Toggle current tool occluded (stay on it) |
| `Shift+1` | Mark all Tool 1 components Out of Scene (-1) |
| `Shift+2` | Mark all Tool 2 components Out of Scene (-1) |
| `Shift+A` | Mark ALL components Out of Scene (negative frame) |

### View
| Key | Action |
|-----|--------|
| `F` | Fit image to view |
| `V` | Toggle pose direction overlay |
| `B` | Toggle batch annotation mode |

### SAM Mode
| Key | Action |
|-----|--------|
| `Enter` | Accept current SAM mask |
| `Tab` | Cycle through SAM mask proposals |
| `Escape` | Exit SAM mode / Cancel drawing |
| `A` / `1` | (SAM apply mode) Assign selected segment to Tool 1 |
| `B` / `4` | (SAM apply mode) Assign selected segment to Tool 2 |

### Batch Mode Shortcuts

When batch mode is active, these keys apply within the grid view:

| Key | Action |
|-----|--------|
| `1` | Mark selected frames: Tool 1 Out |
| `2` | Mark selected frames: Tool 2 Out |
| `A` | Select all frames on current page |
| `D` | Deselect all |
| `B` | Mark selected frames as Broken |
| `X` | Mark selected frames as Exclude |
| `R` | Refresh thumbnails |
| `G` | Focus go-to-frame input |
| `N` | Next trial |
| `<-` / `->` | Previous / Next page |
| `Home` / `End` | First / Last page |
| `Escape` | Exit batch mode |

---

## Mouse Controls

| Action | Result |
|--------|--------|
| **Left-click** | Place polygon vertex / line point / keypoint |
| **Scroll wheel** | Zoom in/out |
| **Right-drag** | Pan view |
| **Middle-click** | Cancel current drawing |
| **Space + drag** | Pan view (alternative) |

### In Edit Mode
| Action | Result |
|--------|--------|
| **Click vertex** | Select vertex |
| **Drag vertex** | Move vertex |
| **Drag line endpoint** | Adjust line position |
| **Box select** | Select multiple vertices |
| **Click keypoint** | Select keypoint for dragging or deletion |

---

## Annotation Workflow

### Basic Workflow

1. **Select dataset and trial** from the left sidebar dropdowns
2. **Set visibility** for each tool component:
   - `Visible` (1): Tool is visible -> annotation required
   - `Occluded` (0): Tool hidden behind tissue/other tool -> no annotation needed
   - `Out of Scene` (-1): Tool is completely absent from the frame -> no annotation needed
3. **Draw mask polygon**:
   - Select mask tool (press `1` for Tool 1 or `4` for Tool 2)
   - Click to place vertices around the tool outline
   - Close the polygon by clicking near the first point (or press `Enter` with 3+ points)
4. **Draw shaft lines**:
   - Select top line tool (press `2` or `5`)
   - Click mask edge for edge-snap placement, or click freely
   - Repeat for bottom line (press `3` or `6`)
   - The midline is auto-computed
5. **Place keypoints**:
   - Press `7` for Tool 1 Joint (or `9` for Tool 2)
   - Click to place, repeat for EE Tip (`8`/`0`), EE Left (`Q`/`O`), EE Right (`W`/`I`)
6. **Navigate to next frame** (press `->`)

### Quick Workflows

- **Auto-advance**: Enable the checkbox in Options to automatically advance to the next tool after completing each component
- **Quick occluded + next**: Press `N` to mark the current tool as occluded and immediately select the next tool
- **Negative frame**: Press `Shift+A` to mark all components as Out of Scene (both tools absent)
- **Edge-snap for shaft lines**: When a mask exists, clicking near a mask edge snaps the shaft line endpoint to that edge
- **Copy prior frame** (`P`): Copy annotations from the previous frame when tools move slowly

### Batch Annotation Workflow

For quickly labeling many frames (e.g., marking negative/broken frames):

1. Press `B` or click the Batch Mode button to enter grid view
2. **Select frames**: Click thumbnails, Shift+click for range, or press `A` to select all on page
3. **Apply operation**: Press `1`/`2` to mark Tool Out, `B` for broken, `X` for exclude
4. Navigate pages with arrow keys
5. Press `B` or `Escape` to return to normal annotation view

### Skipping Frames

Press `S` to mark a frame as skipped (e.g., blurry, transitional, or problematic). Press `S` again to unskip.

### Editing Annotations

Press `E` to enter edit mode:
- Click and drag vertices to adjust mask shape
- Click and drag line endpoints to adjust shaft lines
- Click and drag keypoints to reposition them
- Use box selection (click and drag) to select multiple vertices
- Press `Delete` to remove selected vertices or keypoints

---

## Visibility States

Each tool has **6 independently tracked** visibility components: mask, lines, joint, ee_tip, ee_left, ee_right.

| State | Value | Meaning |
|-------|-------|---------|
| **Visible** | 1 | Component is visible in frame -> annotation required |
| **Occluded** | 0 | Component is hidden (behind tissue, other tool) -> no annotation needed |
| **Out of Scene** | -1 | Tool is completely absent from the frame -> no annotation needed |

Set visibility using the dropdown menus in the right sidebar before drawing annotations.

**Negative frame**: When all 12 visibility components (6 per tool) are set to -1, the frame is classified as a "negative" frame. Use `Shift+A` to quickly mark a frame as negative.

---

## Frame Status & Progress

Each frame has one of 5 statuses:

| Status | Color | Meaning |
|--------|-------|---------|
| **Completed** | Green | All visible components have annotations |
| **Partial** | Yellow | Some but not all visible components annotated |
| **Skipped** | Gray | Manually skipped (press `S`) |
| **Negative** | Purple | All components marked Out of Scene (-1) |
| **Broken** | Red | Image is unusable, moved to broken/ folder |

The **progress bar** in the left sidebar shows overall completion. Trial dropdowns also show per-trial completion percentages.

---

## Progressive Loading

The tool uses a 3-phase loading architecture for fast trial switching:

1. **Phase 1** (instant): Frame list + first frame image loaded immediately. You can start annotating right away.
2. **Phase 2** (streaming): Annotation files stream in via Server-Sent Events (SSE) with a progress bar. Existing annotations appear as they load.
3. **Phase 3** (complete): All data loaded, accurate progress calculated and displayed.

This means switching trials is near-instant even with hundreds of annotated frames.

---

## Batch Annotation Mode

Press `B` to enter the batch annotation grid view for mass-labeling frames.

**Features:**
- **Thumbnail grid**: 12 frames per page with color-coded status borders
- **Multi-select**: Click, Shift+click for range, or press `A` for all on page
- **Batch operations**: Mark Tool Out (`1`/`2`), Mark Broken (`B`), Mark Exclude (`X`)
- **Pagination**: Arrow keys or page controls for navigating through all frames
- **Go-to-frame**: Press `G` to jump to a specific frame index

**Use cases:**
- Quickly marking all frames where a tool is absent (negative frames)
- Flagging broken/corrupt images in bulk
- Excluding frames from validation sets

Press `B` or `Escape` to return to normal annotation view.

---

## SAM Integration (AI-Assisted Segmentation)

The tool integrates with SAM (Segment Anything Model) for AI-assisted mask creation.

### Apply Existing (Pre-computed)

1. Click **"Apply Existing"** in the SAM panel
2. Pre-computed segments are displayed as colored overlays
3. **Click segments** to select (toggle selection with multiple clicks)
4. Press `1`/`A` or click **"Tool 1"** to assign selected segments to Tool 1 mask
5. Press `4`/`B` or click **"Tool 2"** to assign to Tool 2 mask
6. Press `Escape` to exit SAM mode

### New SAM (Interactive)

1. Click **"New SAM"** in the SAM panel
2. Click on the image to place prompt points
3. SAM generates mask proposals
4. Press `Tab` to cycle through proposals
5. Press `Enter` to accept the current mask
6. Assign to a tool using the buttons

---

## Data Format

Annotations are stored in JSON format. Each frame annotation contains:

```json
{
  "frame_idx": 0,
  "tool1_mask": [[x1, y1], [x2, y2], ...],
  "tool2_mask": [[x1, y1], [x2, y2], ...],
  "tool1_lines": {
    "top": [[x1, y1], [x2, y2]],
    "bottom": [[x1, y1], [x2, y2]]
  },
  "tool2_lines": {
    "top": [[x1, y1], [x2, y2]],
    "bottom": [[x1, y1], [x2, y2]]
  },
  "tool1_joint": [x, y],
  "tool1_ee_tip": [x, y],
  "tool1_ee_left": [x, y],
  "tool1_ee_right": [x, y],
  "tool2_joint": [x, y],
  "tool2_ee_tip": [x, y],
  "tool2_ee_left": [x, y],
  "tool2_ee_right": [x, y],
  "tool1_visibility": {
    "mask": 1, "lines": 1,
    "joint": 1, "ee_tip": 1, "ee_left": 1, "ee_right": 1
  },
  "tool2_visibility": {
    "mask": 1, "lines": 1,
    "joint": 1, "ee_tip": 1, "ee_left": 1, "ee_right": 1
  },
  "skipped": false,
  "broken": false,
  "exclude": false,
  "last_modified": "2026-02-10T10:30:00"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `frame_idx` | int | Frame number in the video |
| `tool1_mask` / `tool2_mask` | array | Polygon vertices as [x, y] pairs |
| `tool1_lines` / `tool2_lines` | object | Top and bottom shaft lines (2 points each) |
| `tool1_joint` / `tool2_joint` | [x,y] | Joint keypoint (where shaft meets end-effector) |
| `tool1_ee_tip` / `tool2_ee_tip` | [x,y] | End-effector tip keypoint |
| `tool1_ee_left` / `tool2_ee_left` | [x,y] | Left gripper point keypoint |
| `tool1_ee_right` / `tool2_ee_right` | [x,y] | Right gripper point keypoint |
| `tool1_visibility` / `tool2_visibility` | object | Per-component visibility: 6 fields (mask, lines, joint, ee_tip, ee_left, ee_right). Values: 1=visible, 0=occluded, -1=out of scene |
| `skipped` | bool | Whether frame was marked as skipped |
| `broken` | bool | Image is unusable (moved to broken/ folder) |
| `exclude` | bool | Exclude from validation (still usable for training) |
| `last_modified` | string | ISO timestamp of last modification |

### Negative Frames

A frame is "negative" when all 12 visibility components (6 per tool) are set to -1. This means both tools are completely absent. Negative frames are valid annotations (confirming no tools are present) but don't require drawing any geometry.

---

## Output Location

Annotations are stored as **per-frame JSON files** in a directory per trial:

```
outputs/annotations/
+-- 7DOF2024_Trial1/
|   +-- frame_0000.json      <- individual frame annotation
|   +-- frame_0100.json
|   +-- _progress.json       <- cached progress summary
|   +-- ...
+-- 7DOF2024_Trial1.json     <- legacy monolithic file (auto-migrated)
+-- backups/
    +-- 7DOF2024_Trial1_20260210_120000.json   <- datetime-stamped backup
```

- **Per-frame files**: Each frame gets its own JSON file to prevent data loss from concurrent writes
- **Legacy monolithic files**: Old single-file annotations are auto-migrated to per-frame format on first load
- **Progress cache**: `_progress.json` stores precomputed completion stats for fast dashboard rendering
- **Backups**: Created on demand via the API; stored with datetime stamps

**Auto-save**: Annotations are saved immediately on every change. No manual save required (or press `Ctrl+S` to force save).

---

## Frame Sampling

To reduce annotation workload, the tool samples frames at configurable intervals:

| Option | Description |
|--------|-------------|
| Every 25 | Annotate every 25th frame |
| Every 50 | Annotate every 50th frame |
| **Every 100** | Default: Annotate every 100th frame |
| All frames | Annotate every frame (dense annotation) |

Change the sample rate using the dropdown in the Navigation panel. The progress bar updates to reflect the new total.

---

## Status Checklist

The status panel shows annotation completion for the current frame:

- Unchecked: Component not yet annotated
- Checked: Component annotation complete
- Grayed out: Component marked occluded or out of scene

**Clicking a status item** selects the corresponding tool for drawing.

Items are automatically checked when annotations are complete, or grayed out when marked as occluded/out of scene.

---

## Tips & Best Practices

### Efficient Annotation

1. **Use keyboard shortcuts** - Number keys (1-9, 0, Q, W, O, I) are faster than clicking buttons
2. **Copy prior frame** (`P`) when tools move slowly between frames
3. **Use auto-advance** to automatically move to the next tool after completing each component
4. **Press `N`** to quickly mark a tool as occluded and move on
5. **Press `Shift+A`** for negative frames (both tools absent)
6. **Use batch mode** (`B`) for bulk marking of negative/broken frames
7. **Use SAM** for initial masks, then refine with edit mode
8. **Use edge-snap** for shaft lines when a mask already exists

### Quality Guidelines

1. **Mask polygons**: Trace the visible tool boundary accurately
2. **Shaft lines**: Place along the actual edges, not the center
3. **Keypoints**: Place at anatomical landmarks (joint, gripper tips)
4. **Visibility**: Mark occluded when >50% of the component is hidden; mark out of scene when the tool is completely absent
5. **Consistency**: Maintain similar annotation style across frames

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Can't draw | Check that a tool is selected (indicator shows current tool) |
| Annotations not saving | Check browser console for errors; try `Ctrl+S` to force save |
| SAM not working | Ensure SAM model is loaded (check SAM panel status) |
| Image not loading | Verify trial data exists in dataset folder |
| Slow trial switching | Progressive loading should show Phase 1 instantly; check network |

---

## Related Files

| File | Description |
|------|-------------|
| `surgical_annotator/__main__.py` | CLI entry point with `--data-dir`, `--port`, `--host` |
| `surgical_annotator/app.py` | Flask app setup and warmup |
| `surgical_annotator/routes.py` | Flask API routes (frame serving, annotations, phase bulk) |
| `surgical_annotator/phase_routes.py` | Phase annotation/validation tool routes |
| `surgical_annotator/annotation_store.py` | Per-frame JSON storage with auto-save and backups |
| `surgical_annotator/frame_manager.py` | Trial discovery, frame lookup, and sampling |
| `surgical_annotator/phase_definitions.py` | Phase definitions for peg transfer task |
| `surgical_annotator/sam_segmentation.py` | SAM model wrapper for AI segmentation |
| `surgical_annotator/static/js/main.js` | Frontend application logic |
| `surgical_annotator/static/js/phase.js` | Phase annotation tool frontend |
| `surgical_annotator/static/index.html` | Main annotation interface |
| `surgical_annotator/static/phase.html` | Phase annotation interface |
| `surgical_annotator/export_yolo.py` | Export annotations to YOLO format |
| `surgical_annotator/config.py` | Base path config (`AILET_DATA_DIR` env var support) |

---

## Dependencies

### Minimal (annotation only — macOS/offline)

```
flask>=3.0.0
numpy>=1.21.0
Pillow>=9.0.0
```

### Full (with SAM, training, etc. — Windows)

See `requirements.txt` in the project root.

---

## Export Formats

Annotations can be exported for model training:

```bash
# Export to YOLO format
python -m surgical_annotator.export_yolo --output outputs/yolo_dataset
```

---

## Video Phase Annotation

Access at `http://localhost:5000` and open the video mode for a trial. Features:

- **Timeline clips**: Mark frame ranges with phase labels (reach, grasp, transfer, etc.)
- **Copy/Apply/Restore**: Copy annotations as JSON, paste on another machine, restore from localStorage backup
- **Keyboard shortcuts**: `Z/X/C/D/R/T/G` for phase assignment, `Space` play/pause, `S` save
- **Batched saves**: All clips saved in a single request (no 504 timeouts)

---

*Last updated: 2026-04-23*
