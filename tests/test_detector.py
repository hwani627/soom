"""Detector tests using synthetic flow signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import detector


def _make_flow(duration_sec: int = 600, fs: int = 4, apnea_at: tuple[int, int] | None = None) -> pd.DataFrame:
    """Synthesize a sinusoidal breathing signal with optional apnea (zero amplitude) window."""
    n = duration_sec * fs
    t = np.arange(n) / fs
    flow = 30.0 * np.sin(2 * np.pi * 0.25 * t)  # 0.25 Hz ~ 15 breaths/min, ~30 L/min amp
    if apnea_at is not None:
        s, e = apnea_at
        mask = (t >= s) & (t < e)
        flow[mask] = 0.0
    ts = pd.date_range("2026-03-14 22:00:00", periods=n, freq=f"{int(1000/fs)}ms")
    return pd.DataFrame({"Timestamp_ET": ts, "Breathing_Lpm": flow})


def test_resmed_detects_clean_apnea():
    df = _make_flow(duration_sec=600, fs=4, apnea_at=(300, 320))  # 20s apnea after 5min baseline
    events = detector.detect_events(df, detector.PRESETS["resmed"])
    apneas = events[events["type"] == "apnea"]
    assert len(apneas) >= 1, f"expected at least 1 apnea, got {len(apneas)}\n{events}"
    # the detected apnea should overlap [t0+300s, t0+320s]
    e = apneas.iloc[0]
    assert e["duration_sec"] >= 10


def test_no_events_on_clean_signal():
    df = _make_flow(duration_sec=600, fs=4, apnea_at=None)
    events = detector.detect_events(df, detector.PRESETS["resmed"])
    assert len(events) == 0, f"expected 0 events on clean signal, got {len(events)}"


def test_empty_input_returns_empty_frame():
    out = detector.detect_events(pd.DataFrame(), detector.PRESETS["resmed"])
    assert out.empty
    assert list(out.columns) == ["start_ts", "end_ts", "type", "duration_sec", "ratio_min", "method"]
