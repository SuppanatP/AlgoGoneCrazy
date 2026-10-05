"""Convenience entry point for 30-minute market-data aggregation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_data.aggregation import main  # noqa: E402


if __name__ == "__main__":
    main()
