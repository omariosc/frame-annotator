"""CLI entry point for frame-annotator."""

import argparse
import sys
from pathlib import Path

from .app import create_app
from .config import get_default_config, load_config


def main():
    parser = argparse.ArgumentParser(
        prog="frame-annotator",
        description="A lightweight web tool for annotating video frames with classification labels.",
    )
    parser.add_argument(
        "image_dir",
        help="Path to directory containing image frames",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Path to YAML/JSON config file defining annotation classes (default: built-in binary classifier)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for annotations (default: <image_dir>/annotations/)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5001,
        help="Port number (default: 5001)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"Error: '{image_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            print(f"Error: config file '{config_path}' not found", file=sys.stderr)
            sys.exit(1)
        config = load_config(config_path)
    else:
        config = get_default_config()

    app = create_app(image_dir, config, output_dir=args.output)

    print("Starting frame-annotator...")
    print(f"  Project: {config['project']['name']}")
    print(f"  Images:  {image_dir.resolve()}")
    print(f"  Config:  {args.config or 'built-in default'}")
    print(f"  Open http://{args.host}:{args.port} in your browser")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
