"""Tests for src/hypnogram.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import hypnogram

REPO = Path(__file__).resolve().parents[1]
SYNTH_HYP = REPO / "tests" / "fixtures" / "synth_hypnogram.edf"


def test_stage_enum_covers_aasm_five_stages():
    names = {s.name for s in hypnogram.Stage}
    assert {"W", "N1", "N2", "N3", "REM"}.issubset(names)


def test_load_hypnogram_returns_one_w_epoch():
    epochs = hypnogram.load_hypnogram(SYNTH_HYP)
    assert len(epochs) == 1
    e = epochs[0]
    assert e.start_sec == 0.0
    assert e.duration_sec == 30.0
    assert e.stage == hypnogram.Stage.W


def test_to_stage_series_truncates_to_total_dur():
    eps = [
        hypnogram.HypnogramEpoch(0.0, 60.0, hypnogram.Stage.W),
        hypnogram.HypnogramEpoch(60.0, 60.0, hypnogram.Stage.N1),
    ]
    df = hypnogram.to_stage_series(eps, total_dur_sec=90.0)
    assert isinstance(df, pd.DataFrame)
    assert df["t_sec"].max() == 89.0
    # First minute should all be W; last 30 s should all be N1.
    assert (df.loc[df["t_sec"] < 60, "stage"] == "W").all()
    assert (df.loc[df["t_sec"] >= 60, "stage"] == "N1").all()


def test_to_stage_series_handles_empty_epoch_list():
    df = hypnogram.to_stage_series([], total_dur_sec=10.0)
    assert (df["stage"] == "UNK").all()
    assert len(df) == 10
