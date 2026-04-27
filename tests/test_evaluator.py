"""Evaluator tests with synthetic events / SpO2 inputs."""
from __future__ import annotations

import pandas as pd

from src import evaluator


def _events_df(types: list[str], starts: list[str], ends: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "start_ts": pd.to_datetime(starts), "end_ts": pd.to_datetime(ends),
        "event_type": types, "tooltip": [""] * len(types),
    })


def _spo2_df(values: list[float], freq_sec: int = 1) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame({
        "Timestamp_ET": pd.date_range("2026-04-14 00:00:00", periods=n, freq=f"{freq_sec}s"),
        "SpO2_pct": values,
    })


def test_ahi_grading_thresholds():
    pass_evs = _events_df(["CA"] * 3, ["2026-04-14 00:00:00"] * 3, ["2026-04-14 00:00:11"] * 3)
    warn_evs = _events_df(["CA"] * 8, ["2026-04-14 00:00:00"] * 8, ["2026-04-14 00:00:11"] * 8)
    fail_evs = _events_df(["CA"] * 20, ["2026-04-14 00:00:00"] * 20, ["2026-04-14 00:00:11"] * 20)

    assert evaluator._ahi_card(pass_evs, 1.0)["grade"] == "pass"
    assert evaluator._ahi_card(warn_evs, 1.0)["grade"] == "warn"
    assert evaluator._ahi_card(fail_evs, 1.0)["grade"] == "fail"


def test_leak_card_threshold():
    s_pass = pd.DataFrame({"LeakRate_Lpm": [5] * 100})
    s_fail = pd.DataFrame({"LeakRate_Lpm": [50] * 100})
    assert evaluator._leak_card(s_pass)["grade"] == "pass"
    assert evaluator._leak_card(s_fail)["grade"] == "fail"


def test_usage_card_threshold():
    assert evaluator._usage_card(5.0)["grade"] == "pass"
    assert evaluator._usage_card(3.0)["grade"] == "fail"


def test_lowest_spo2_card():
    assert evaluator._lowest_spo2_card(_spo2_df([95, 96, 97]))["grade"] == "pass"
    assert evaluator._lowest_spo2_card(_spo2_df([88, 96, 97]))["grade"] == "warn"
    assert evaluator._lowest_spo2_card(_spo2_df([82, 96, 97]))["grade"] == "fail"


def test_t90_card_thresholds():
    healthy = _spo2_df([97] * 3600)
    bad     = _spo2_df([85] * 3600)
    assert evaluator._t90_card(healthy)["grade"] == "pass"
    assert evaluator._t90_card(bad)["grade"] == "fail"


def test_compare_timestamp_overlap():
    detected = pd.DataFrame({
        "start_ts": pd.to_datetime(["2026-04-14 00:00:00", "2026-04-14 01:00:00"]),
        "end_ts":   pd.to_datetime(["2026-04-14 00:00:15", "2026-04-14 01:00:15"]),
        "type":     ["apnea", "apnea"],
    })
    sleephq = _events_df(
        ["CA", "H"],
        ["2026-04-14 00:00:05", "2026-04-14 02:00:00"],
        ["2026-04-14 00:00:20", "2026-04-14 02:00:10"],
    )
    out = evaluator.compare_with_sleephq(detected, sleephq)
    # detected[0] overlaps sleephq[0] → TP, detected[1] no overlap → FP
    assert out["detected_count"] == 2
    assert out["sleephq_count"] == 2
    assert out["precision"] == 0.5    # 1/2
    assert out["recall"]    == 0.5    # 1/2
