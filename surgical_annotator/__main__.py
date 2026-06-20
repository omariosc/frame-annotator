"""Entry point for running the surgical annotator as a module.

Usage:
    # Point it at your data directory:
    python -m surgical_annotator --data-dir /path/to/data
    # or, after `pip install -e .`:
    surgical-annotator --data-dir /path/to/data

    The data directory should contain any of:
        6DOF2023/    7DOF2024/    BAPES2024/    outputs/
    (missing datasets are skipped). Opens at http://localhost:5000.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Surgical Image Annotation Tool")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root data directory containing 6DOF2023/, 7DOF2024/, BAPES2024/, outputs/. "
             "Defaults to ./data if omitted.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    # Set env var BEFORE importing anything that reads config
    if args.data_dir:
        os.environ["AILET_DATA_DIR"] = args.data_dir

    # Now import and run
    from .app import app, main as app_main
    app_main(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
