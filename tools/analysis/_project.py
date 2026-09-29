"""Project-root bootstrap for standalone analysis scripts."""

from pathlib import Path
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Legacy analysis scripts use repository-relative data and output paths.
os.chdir(PROJECT_ROOT)
