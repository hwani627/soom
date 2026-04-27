"""Smoke tests for src/edf_plotting.py — verify Figures build without error."""
from __future__ import annotations

from pathlib import Path

from src import edf_loader, edf_plotting, hypnogram

REPO = Path(__file__).resolve().parents[1]
SYNTH_PSG = REPO / "tests" / "fixtures" / "synth_psg.edf"
SYNTH_HYP = REPO / "tests" / "fixtures" / "synth_hypnogram.edf"


def test_freezoom_figure_builds_without_hypnogram():
    meta = edf_loader.load_meta(SYNTH_PSG)
    signals = {
        0: edf_loader.load_signal(SYNTH_PSG, 0),
        1: edf_loader.load_signal(SYNTH_PSG, 1),
    }
    fig = edf_plotting.make_freezoom_figure(
        meta, signals, hypno=None, channels=[0, 1],
    )
    assert fig.data, "expected at least one trace"


def test_freezoom_figure_builds_with_hypnogram():
    meta = edf_loader.load_meta(SYNTH_PSG)
    signals = {0: edf_loader.load_signal(SYNTH_PSG, 0)}
    hypno = hypnogram.load_hypnogram(SYNTH_HYP)
    fig = edf_plotting.make_freezoom_figure(
        meta, signals, hypno=hypno, channels=[0],
    )
    assert fig.data
