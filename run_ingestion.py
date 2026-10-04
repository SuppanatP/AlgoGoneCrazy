"""Convenience entry point for the source-layout package."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_data.pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
