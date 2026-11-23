"""Download the IBM Telco Customer Churn dataset (Apache-2.0) into data/raw/."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Dataset already present: {DEST}")
        return 0
    try:
        urllib.request.urlretrieve(URL, DEST)  # noqa: S310 - fixed https URL
    except OSError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved dataset to {DEST} ({DEST.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
