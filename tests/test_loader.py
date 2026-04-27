"""Loader tests for SleepHQ long-format CSVs."""
from __future__ import annotations

from pathlib import Path

from src import loader

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "CAPA_Data"


def test_list_dates_returns_seven_days():
    dates = loader.list_available_dates(DATA)
    assert len(dates) == 7
    assert dates[0] == "2026-04-13" and dates[-1] == "2026-04-19"


def test_load_session_has_core_channels():
    session = loader.load_session(DATA, "2026-04-14")
    for key in ("breathing", "spo2", "pulserate", "leakrate", "events"):
        assert key in session, f"missing {key}"
    assert "Breathing_Lpm" in session["breathing"].columns
    assert "Timestamp_ET" in session["breathing"].columns
    assert "SpO2_pct" in session["spo2"].columns
    assert {"start_ts", "end_ts", "event_type"}.issubset(session["events"].columns)


def test_session_duration_is_positive():
    session = loader.load_session(DATA, "2026-04-14")
    h = loader.session_duration_hours(session)
    assert h is not None and 0.5 < h < 14, f"unexpected duration: {h}"


def test_sleep_stage_loaded():
    session = loader.load_session(DATA, "2026-04-14")
    assert "sleep_stage" in session
    assert "sleep_stage" in session["sleep_stage"].columns
