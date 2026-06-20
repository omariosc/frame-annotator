"""Configuration for the surgical annotator (standalone).

The data directory is read from the ``AILET_DATA_DIR`` environment variable,
which the CLI (``surgical-annotator --data-dir ...`` or
``python -m surgical_annotator --data-dir ...``) sets before this module is
imported. It should contain ``6DOF2023/``, ``7DOF2024/``, ``BAPES2024/`` and
``outputs/`` subdirectories (any missing datasets are skipped).

If ``--data-dir`` is not supplied it defaults to ``./data`` relative to the
current working directory.
"""

import os
from pathlib import Path

# Override with --data-dir (which sets AILET_DATA_DIR). Defaults to ./data.
BASE_DIR = Path(os.environ.get("AILET_DATA_DIR", str(Path.cwd() / "data")))
DATA_6DOF = BASE_DIR / "6DOF2023"
DATA_7DOF = BASE_DIR / "7DOF2024"
DATA_BAPES = BASE_DIR / "BAPES2024"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Create output directories only if the base data dir actually exists.
if BASE_DIR.exists():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
