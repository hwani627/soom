"""SleepHQ long-format CSV loader.

Three input files, all timestamped in UTC:
  1. sleephq_rawdata_*.csv          — long format: date, timestamp_ms, datetime_utc, data_type, value
  2. sleephq_events_AHI_*.csv       — events: date, start/end_timestamp_ms, ..., event_type, tooltip
  3. sleephq_sleep_stages_*.csv     — date, timestamp_ms, datetime_utc, sleep_stage

The loader splits the multi-day raw data into per-date sessions and pivots the
long format into per-channel DataFrames keyed by canonical channel names.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# Canonical channel keys (used by detector/evaluator/plotting).
CHANNELS = (
    "breathing", "pressure", "epap", "leakrate", "flowlimit", "snore",
    "spo2", "pulserate", "movement",
    "resprate", "minutevent", "tidalvolume",
)

# Map SleepHQ data_type strings to canonical keys.
DATA_TYPE_MAP = {
    "Breathing":         ("breathing",   "Breathing_Lpm"),
    "Pressure_Pressure": ("pressure",    "Pressure_cmH2O"),
    "Pressure_EPAP":     ("epap",        "EPAP_cmH2O"),
    "LeakRate":          ("leakrate",    "LeakRate_Lpm"),
    "FlowLimit":         ("flowlimit",   "FlowLimit"),
    "Snore":             ("snore",       "Snore"),
    "SpO2":              ("spo2",        "SpO2_pct"),
    "PulseRate":         ("pulserate",   "PulseRate_bpm"),
    "Movement":          ("movement",    "Movement_idx"),
    "RespRate":          ("resprate",    "RespRate_bpm"),
    "MinuteVent":        ("minutevent",  "MinuteVent_Lpm"),
    "TidalVolume":       ("tidalvolume", "TidalVolume_mL"),
}


def list_available_dates(data_root: Path) -> list[str]:
    """Return sorted unique date strings (YYYY-MM-DD) found in the rawdata file."""
    raw_csv = _find_one(data_root, "sleephq_rawdata_*.csv")
    if raw_csv is None:
        return []
    # Read only the date column to keep it cheap.
    df = pd.read_csv(raw_csv, usecols=["date"])
    return sorted(df["date"].dropna().unique().tolist())


def load_session(data_root: Path, date: str) -> dict[str, pd.DataFrame]:
    """Load one date's session as a dict of per-channel DataFrames.

    Returns
    -------
    dict with keys from CHANNELS plus optional:
        - 'events':      DataFrame of SleepHQ events for that date
        - 'sleep_stage': DataFrame of sleep stage transitions for that date
    """
    raw_csv    = _find_one(data_root, "sleephq_rawdata_*.csv")
    events_csv = _find_one(data_root, "sleephq_events_AHI_*.csv")
    stages_csv = _find_one(data_root, "sleephq_sleep_stages_*.csv")
    if raw_csv is None:
        raise FileNotFoundError(f"sleephq_rawdata_*.csv not found in {data_root}")

    raw = pd.read_csv(raw_csv)
    raw = raw[raw["date"] == date].copy()
    raw["datetime_utc"] = pd.to_datetime(raw["datetime_utc"], errors="coerce")
    raw = raw.dropna(subset=["datetime_utc"])

    out: dict[str, pd.DataFrame] = {}
    for data_type, (key, value_col) in DATA_TYPE_MAP.items():
        sub = raw[raw["data_type"] == data_type]
        if sub.empty:
            continue
        df = pd.DataFrame({
            "Timestamp_ET": sub["datetime_utc"].values,  # column name kept for backwards-compat
            value_col: pd.to_numeric(sub["value"], errors="coerce").values,
        }).dropna().sort_values("Timestamp_ET").reset_index(drop=True)
        out[key] = df

    if events_csv is not None:
        ev = pd.read_csv(events_csv)
        ev = ev[ev["date"] == date].copy()
        ev["start_ts"] = pd.to_datetime(ev["start_datetime_utc"], errors="coerce")
        ev["end_ts"]   = pd.to_datetime(ev["end_datetime_utc"],   errors="coerce")
        ev = ev.dropna(subset=["start_ts", "end_ts"]).reset_index(drop=True)
        out["events"] = ev[["start_ts", "end_ts", "event_type", "tooltip"]]

    if stages_csv is not None:
        st = pd.read_csv(stages_csv)
        st = st[st["date"] == date].copy()
        st["Timestamp_ET"] = pd.to_datetime(st["datetime_utc"], errors="coerce")
        st = st.dropna(subset=["Timestamp_ET"]).sort_values("Timestamp_ET").reset_index(drop=True)
        out["sleep_stage"] = st[["Timestamp_ET", "sleep_stage"]]

    return out


def session_duration_hours(session: dict[str, pd.DataFrame]) -> Optional[float]:
    br = session.get("breathing")
    if br is None or br.empty:
        return None
    delta = br["Timestamp_ET"].iloc[-1] - br["Timestamp_ET"].iloc[0]
    return delta.total_seconds() / 3600.0


def _find_one(root: Path, pattern: str) -> Optional[Path]:
    matches = sorted(Path(root).glob(pattern))
    return matches[0] if matches else None
