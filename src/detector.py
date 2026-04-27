"""Apnea / Hypopnea detector on Breathing flow signal.

Two preset methods:
  - 'aasm':   Apnea = >=90% drop, Hypopnea = >=30% drop, baseline=2 min mean
              (SpO2 desat condition NOT verified; flagged in UI)
  - 'resmed': Apnea = <25% of baseline, Hypopnea = <50% of baseline,
              baseline = 100 s rolling 60th percentile (ResMed AutoSet style)

Limitations
-----------
* CA (Clear Airway) vs OA (Obstructive) cannot be distinguished here -- requires
  Forced Oscillation Technique (FOT) channel which is not in the export CSV.
* AASM hypopnea per AASM 2012 also requires SpO2 desat >=3% or arousal -- absent.

References
----------
- Berry RB et al., J Clin Sleep Med 2012; 8(5): 597-619.
- ResMed AirSense 10 AutoSet Algorithm White Paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Method = Literal["aasm", "resmed"]


@dataclass(frozen=True)
class DetectorParams:
    method: Method = "resmed"
    apnea_thr: float = 0.25
    hypop_thr: float = 0.50
    min_duration_sec: float = 10.0
    baseline_window_sec: float = 100.0


PRESETS: dict[Method, DetectorParams] = {
    "resmed": DetectorParams("resmed", 0.25, 0.50, 10.0, 100.0),
    "aasm":   DetectorParams("aasm",   0.10, 0.70, 10.0, 120.0),
}


def _flow_envelope(flow: np.ndarray, fs_hz: float, window_sec: float = 2.0) -> np.ndarray:
    """Rolling RMS of flow signal as amplitude envelope."""
    win = max(int(round(window_sec * fs_hz)), 1)
    sq = np.square(flow.astype(float))
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    means = (csum[win:] - csum[:-win]) / win
    pad = np.full(win - 1, means[0] if means.size else 0.0)
    rms = np.sqrt(np.maximum(np.concatenate([pad, means]), 0.0))
    return rms


def _baseline(envelope: np.ndarray, fs_hz: float, window_sec: float, method: Method) -> np.ndarray:
    """Baseline amplitude. ResMed: rolling 60th pct, AASM: rolling mean of last 2 min."""
    win = max(int(round(window_sec * fs_hz)), 1)
    s = pd.Series(envelope)
    if method == "aasm":
        return s.rolling(win, min_periods=max(win // 4, 1)).mean().bfill().to_numpy()
    return s.rolling(win, min_periods=max(win // 4, 1)).quantile(0.60).bfill().to_numpy()


def _estimate_fs(timestamps: pd.Series) -> float:
    """Median sampling frequency from timestamp diffs (Hz)."""
    if len(timestamps) < 2:
        return 1.0
    deltas = timestamps.diff().dt.total_seconds().dropna()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return 1.0
    return float(1.0 / deltas.median())


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx_inclusive) for True-runs in mask."""
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def detect_events(breathing: pd.DataFrame, params: DetectorParams) -> pd.DataFrame:
    """Detect Apnea / Hypopnea candidate intervals.

    Parameters
    ----------
    breathing : DataFrame with columns ['Timestamp_ET', 'Breathing_Lpm']
    params : DetectorParams

    Returns
    -------
    DataFrame columns:
        start_ts, end_ts, type, duration_sec, ratio_min, method
    """
    if breathing is None or breathing.empty:
        return _empty_events_frame()

    ts = breathing["Timestamp_ET"]
    flow = breathing["Breathing_Lpm"].to_numpy()
    fs = _estimate_fs(ts)
    if fs <= 0 or not np.isfinite(fs):
        return _empty_events_frame()

    envelope = _flow_envelope(flow, fs_hz=fs)
    base = _baseline(envelope, fs_hz=fs, window_sec=params.baseline_window_sec, method=params.method)
    base_safe = np.where(base > 1e-6, base, 1e-6)
    ratio = envelope / base_safe

    min_samples = max(int(round(params.min_duration_sec * fs)), 1)
    apnea_mask = ratio < params.apnea_thr
    hypop_mask = (ratio >= params.apnea_thr) & (ratio < params.hypop_thr)

    events: list[dict] = []
    for label, mask in (("apnea", apnea_mask), ("hypopnea", hypop_mask)):
        for s, e in _runs(mask):
            if (e - s + 1) < min_samples:
                continue
            events.append({
                "start_ts": ts.iloc[s],
                "end_ts": ts.iloc[e],
                "type": label,
                "duration_sec": float((ts.iloc[e] - ts.iloc[s]).total_seconds()),
                "ratio_min": float(np.min(ratio[s:e + 1])),
                "method": params.method,
            })

    df = pd.DataFrame(events, columns=[
        "start_ts", "end_ts", "type", "duration_sec", "ratio_min", "method"
    ])
    return df.sort_values("start_ts").reset_index(drop=True)


def _empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["start_ts", "end_ts", "type", "duration_sec", "ratio_min", "method"])
