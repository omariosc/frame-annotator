"""Flask application factory for frame-annotator."""

import csv
import json
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory


def create_app(image_dir, config, output_dir=None):
    """Create and configure the Flask annotation app.

    Args:
        image_dir: Path to directory containing image frames.
        config: Validated config dict from config.py.
        output_dir: Directory for annotation output. Defaults to image_dir/annotations.
    """
    image_dir = Path(image_dir).resolve()
    if output_dir is None:
        output_dir = image_dir / "annotations"
    else:
        output_dir = Path(output_dir).resolve()

    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))

    pattern = config.get("images", {}).get("pattern", "*.png")
    frames = sorted([f.name for f in image_dir.glob(pattern)])
    total_frames = len(frames)

    if total_frames == 0:
        print(f"Warning: No images matching '{pattern}' found in {image_dir}")

    annotations_file = output_dir / "annotations.json"
    annotations_csv = output_dir / "annotations.csv"

    @app.route("/")
    def index():
        return render_template(
            "annotator.html", total_frames=total_frames, config=config
        )

    @app.route("/api/frames")
    def get_frames():
        return jsonify({"frames": frames, "total": total_frames})

    @app.route("/images/<filename>")
    def serve_image(filename):
        return send_from_directory(str(image_dir), filename)

    @app.route("/api/load_annotations")
    def load_annotations():
        if annotations_file.exists():
            with open(annotations_file, "r") as f:
                return jsonify(json.load(f))
        return jsonify({"clips": []})

    @app.route("/api/save_annotations", methods=["POST"])
    def save_annotations():
        data = request.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main JSON
        with open(annotations_file, "w") as f:
            json.dump(data, f, indent=2)

        # Timestamped backup
        backup_json = output_dir / f"annotations_{timestamp}.json"
        with open(backup_json, "w") as f:
            json.dump(data, f, indent=2)

        # Export CSV
        _export_csv(data, annotations_csv, frames)

        backup_csv = output_dir / f"annotations_{timestamp}.csv"
        _export_csv(data, backup_csv, frames)

        return jsonify(
            {
                "success": True,
                "message": f"Annotations saved successfully (backup: {timestamp})",
            }
        )

    return app


def _export_csv(data, output_path, frames):
    """Export annotations to CSV format."""
    clips = data.get("clips", [])
    rows = []

    for clip_idx, clip in enumerate(clips):
        start = clip["start"]
        end = clip["end"]
        cls = clip["class"]

        for frame_num in range(start, end + 1):
            rows.append(
                {
                    "frame": frame_num,
                    "filename": frames[frame_num] if frame_num < len(frames) else "",
                    "clip_id": clip_idx,
                    "class": cls,
                }
            )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "filename", "clip_id", "class"])
        writer.writeheader()
        writer.writerows(rows)
