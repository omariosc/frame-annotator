"""Configuration loading and validation for frame-annotator."""

import json
import re
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "project": {
        "name": "Frame Annotator",
        "description": "Annotate video frames with classification labels",
    },
    "images": {
        "pattern": "*.png",
    },
    "classes": [
        {
            "id": "positive",
            "name": "Positive",
            "color": "#28a745",
            "shortcut": "1",
            "description": "Positive class",
        },
        {
            "id": "negative",
            "name": "Negative",
            "color": "#dc3545",
            "shortcut": "2",
            "description": "Negative class",
        },
    ],
}

HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def load_config(config_path):
    """Load annotation config from a YAML or JSON file.

    Returns a validated config dict.
    """
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        config = yaml.safe_load(text)
    elif path.suffix == ".json":
        config = json.loads(text)
    else:
        # Try YAML first, then JSON
        try:
            config = yaml.safe_load(text)
        except Exception:
            config = json.loads(text)

    return _validate(config)


def get_default_config():
    """Return the built-in default config."""
    return _validate(DEFAULT_CONFIG)


def _validate(config):
    """Validate and normalize a config dict."""
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML/JSON object")

    # Defaults
    config.setdefault("project", {})
    config["project"].setdefault("name", "Frame Annotator")
    config["project"].setdefault("description", "")
    config.setdefault("images", {})
    config["images"].setdefault("pattern", "*.png")

    classes = config.get("classes")
    if not classes or not isinstance(classes, list):
        raise ValueError("Config must define at least one class in 'classes'")

    seen_ids = set()
    seen_shortcuts = set()

    for cls in classes:
        # Required fields
        for field in ("id", "name", "color"):
            if field not in cls:
                raise ValueError(f"Class missing required field '{field}': {cls}")

        cls_id = str(cls["id"])
        cls["id"] = cls_id

        if cls_id in seen_ids:
            raise ValueError(f"Duplicate class id: '{cls_id}'")
        seen_ids.add(cls_id)

        if not HEX_COLOR_RE.match(cls["color"]):
            raise ValueError(f"Invalid hex color '{cls['color']}' for class '{cls_id}'")

        shortcut = cls.get("shortcut")
        if shortcut:
            if shortcut in seen_shortcuts:
                raise ValueError(f"Duplicate shortcut '{shortcut}'")
            seen_shortcuts.add(shortcut)

        cls.setdefault("description", "")

        # Subcategories
        subs = cls.get("subcategories")
        if subs:
            for sub in subs:
                for field in ("id", "name"):
                    if field not in sub:
                        raise ValueError(
                            f"Subcategory missing '{field}' in class '{cls_id}': {sub}"
                        )
                sub_id = str(sub["id"])
                sub["id"] = sub_id
                if len(sub_id) != 1:
                    raise ValueError(
                        f"Subcategory id must be a single character, got '{sub_id}'"
                    )
                sub_shortcut = sub.get("shortcut")
                if sub_shortcut:
                    if sub_shortcut in seen_shortcuts:
                        raise ValueError(f"Duplicate shortcut '{sub_shortcut}'")
                    seen_shortcuts.add(sub_shortcut)

    return config
