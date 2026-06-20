"""Flask application for surgical image annotation tool.

Run with: python -m surgical_annotator.app
"""

import logging
import threading
import time

from pathlib import Path
from flask import Flask, send_from_directory

from .routes import api_bp
from .phase_routes import phase_bp

logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(
    __name__,
    static_folder='static',
    static_url_path='/static'
)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Register API blueprints
app.register_blueprint(api_bp, url_prefix='/api')
app.register_blueprint(phase_bp)

# ── Startup warmup ──────────────────────────────────────────────────────────
# Pre-warm all caches on the first HTTP request.  Uses before_request (not a
# background thread) because Flask debug=True forks via the reloader — a
# thread started in main() would run in the parent process, not the child.
_warmup_complete = False
_warmup_lock = threading.Lock()


@app.before_request
def _ensure_warmup():
    global _warmup_complete
    if _warmup_complete:
        return
    with _warmup_lock:
        if _warmup_complete:
            return
        t0 = time.time()
        logger.info("Warming up caches...")

        from .frame_manager import get_frame_manager
        from .annotation_store import get_annotation_store, ANNOTATION_DIR

        fm = get_frame_manager()
        fm.warmup()  # discover_all_trials + ensure_frame_count for all

        store = get_annotation_store()
        # Seed progress summaries if none exist yet
        has_any = False
        if ANNOTATION_DIR.exists():
            has_any = any(
                (d / '_progress.json').exists()
                for d in ANNOTATION_DIR.iterdir()
                if d.is_dir() and d.name != 'backups'
            )
        if not has_any:
            logger.info("No progress summaries found — seeding from existing annotations")
            store.seed_progress_summaries()

        # Load all summaries into memory
        store.warmup_progress_cache(fm)

        _warmup_complete = True
        logger.info(f"Warmup done in {time.time() - t0:.2f}s")


@app.route('/')
def index():
    """Serve the main annotation interface."""
    return send_from_directory(app.static_folder, 'index.html')


def main(host: str = '0.0.0.0', port: int = 5000):
    """Run the annotation server.

    Args:
        host: Host to bind to.
        port: Port to listen on.
    """
    from surgical_annotator.config import BASE_DIR

    print("=" * 60)
    print("Surgical Image Annotation Tool")
    print("=" * 60)
    print(f"\nData directory: {BASE_DIR}")
    print(f"\nStarting server...")
    print(f"Open http://localhost:{port} in your browser")
    print("Press Ctrl+C to stop\n")

    # Run Flask app
    app.run(host=host, port=port, debug=True)


if __name__ == '__main__':
    main()
