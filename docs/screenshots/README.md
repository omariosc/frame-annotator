# Screenshots

These images are referenced by the top-level `README.md`.

## Present (frame-annotator)
- `overview.png` — the frame-annotator clip-classification interface.
- `surgical_safety.png` — frame-annotator running the `examples/surgical_safety.yaml` config (robotic surgical-safety labelling).

## To add (surgical-annotator)
Capture these from `surgical-annotator --data-dir /path/to/data` (http://localhost:5000) and save them here with these exact names:

- `surgical_annotator_interface.png` — the main annotation view: a frame with two-instrument masks, shaft lines, and keypoints, plus the right-hand visibility / SAM / JSON panels.
- `surgical_annotator_batch.png` — **batch mode** (press `B`): the paginated thumbnail grid with colour-coded status borders and a multi-selection.
- `surgical_annotator_video_phase.png` — **video mode** with the phase/event timeline showing labelled clips.

Recommended: PNG, ~1400px wide, light UI theme for legibility on GitHub.
