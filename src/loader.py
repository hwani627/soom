"""CSV loader for ResMed AirSense 10 / SleepHQ-export style sessions."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

EXPECTED_FILES = {
    "ahi_summary": "AHI_Summary.csv",
    "statistics": "Statistics.csv",
    "pressure": "Pressure.csv",
    "breathing": "Breathing.csv",
    "flowlimit": "FlowLimit.csv",
    "leakrate": "LeakRate.csv",
    "snore": "Snore.csv",
}

TIMESERIES_KEYS = ("pressure", "breathing", "flowlimit", "leakrate", "snore")


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Timestamp_ET" in df.columns:
        df["Timestamp_ET"] = pd.to_datetime(df["Timestamp_ET"], errors="coerce")
        df = df.dropna(subset=["Timestamp_ET"]).sort_values("Timestamp_ET").reset_index(drop=True)
    return df


def load_session(session_dir: Path, date_prefix: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """Load up to 7 CSVs from a date folder.

    Files may be missing (e.g. Statistics.csv on some dates) - missing keys are absent in the result.
    `date_prefix` (e.g. '20260314') prepends to filenames; auto-detected from folder name if None.
    """
    session_dir = Path(session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session dir not found: {session_dir}")

    if date_prefix is None:
        date_prefix = session_dir.name

    out: dict[str, pd.DataFrame] = {}
    for key, fname in EXPECTED_FILES.items():
        candidate = session_dir / f"{date_prefix}_{fname}"
        if not candidate.exists():
            candidate = session_dir / fname
        if candidate.exists():
            out[key] = _read_csv(candidate)
    return out


def session_duration_hours(session: dict[str, pd.DataFrame]) -> Optional[float]:
    """Estimate session duration from breathing timestamps (start to end)."""
    br = session.get("breathing")
    if br is None or br.empty or "Timestamp_ET" not in br.columns:
        return None
    delta = br["Timestamp_ET"].iloc[-1] - br["Timestamp_ET"].iloc[0]
    return delta.total_seconds() / 3600.0


def list_available_dates(data_root: Path) -> list[str]:
    """Return sorted list of date-folder names (YYYYMMDD) under data_root."""
    data_root = Path(data_root)
    if not data_root.exists():
        return []
    return sorted(
        d.name for d in data_root.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
    )
