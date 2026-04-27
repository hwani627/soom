"""Loader tests against the bundled CAPA_Data."""
from __future__ import annotations

from pathlib import Path

import pytest

from src import loader

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "CAPA_Data"


def test_list_dates_returns_13_or_more():
    dates = loader.list_available_dates(DATA)
    assert len(dates) >= 13
    assert all(d.isdigit() and len(d) == 8 for d in dates)


def test_load_session_for_20260314_has_core_files():
    session = loader.load_session(DATA / "20260314")
    for key in ["ahi_summary", "breathing", "leakrate", "flowlimit", "pressure", "snore"]:
        assert key in session, f"missing {key}"
    assert "Breathing_Lpm" in session["breathing"].columns
    assert "Timestamp_ET" in session["breathing"].columns
    assert len(session["breathing"]) > 1000


def test_session_duration_is_positive_for_20260314():
    session = loader.load_session(DATA / "20260314")
    h = loader.session_duration_hours(session)
    assert h is not None and 1 < h < 12, f"unexpected duration: {h}"


def test_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        loader.load_session(DATA / "19000101")
