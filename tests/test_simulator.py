"""Tests for simulator UI helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simulator import ui_state  # noqa: E402
from signal_processing import generator  # noqa: E402


class TestUIState:
    def test_widget_dict_to_scenario_config_roundtrip(self):
        widgets = {
            "duration_s": 300.0, "fs_hz": 100.0,
            "rr_bpm": 15.0, "tv_ml": 500.0, "ie_ratio": 0.5,
            "base_pressure_cmh2o": 9.5,
            "epr_enabled": False, "epr_relief_cmh2o": 0.0,
            "n_oa": 2, "n_ca": 1, "n_ma": 0, "n_hypopnea": 1,
            "n_snore": 1, "n_cough": 0,
            "oa_mean_dur_s": 18.0, "ca_mean_dur_s": 15.0,
            "hypopnea_mean_dur_s": 20.0,
            "hr_bpm": 70.0, "cardiogenic_amplitude_cmh2o": 0.25,
            "measurement_noise_std_cmh2o": 0.05,
            "power_line_50hz_enabled": False,
            "unintentional_leak_lpm": 0.0,
        }
        cfg = ui_state.build_scenario_config(widgets, seed=42)
        assert isinstance(cfg, generator.ScenarioConfig)
        assert cfg.duration_s == 300.0
        assert cfg.rr_bpm == 15.0
        assert cfg.ie_ratio == 0.5
        assert len(cfg.obstructive_apneas) == 2
        assert len(cfg.central_apneas) == 1
        assert len(cfg.hypopneas) == 1
        assert cfg.power_line_50hz_amplitude_cmh2o == 0.0

    def test_event_count_to_schedule_no_overlap(self):
        widgets = {
            "duration_s": 600.0, "fs_hz": 100.0,
            "rr_bpm": 15.0, "tv_ml": 500.0, "ie_ratio": 0.5,
            "base_pressure_cmh2o": 9.5,
            "epr_enabled": False, "epr_relief_cmh2o": 0.0,
            "n_oa": 5, "n_ca": 3, "n_ma": 0, "n_hypopnea": 4,
            "n_snore": 0, "n_cough": 0,
            "oa_mean_dur_s": 18.0, "ca_mean_dur_s": 15.0,
            "hypopnea_mean_dur_s": 20.0,
            "hr_bpm": 70.0, "cardiogenic_amplitude_cmh2o": 0.25,
            "measurement_noise_std_cmh2o": 0.05,
            "power_line_50hz_enabled": False,
            "unintentional_leak_lpm": 0.0,
        }
        cfg = ui_state.build_scenario_config(widgets, seed=7)
        intervals = []
        for s, d in cfg.obstructive_apneas:
            intervals.append((s, s + d))
        for s, d in cfg.central_apneas:
            intervals.append((s, s + d))
        for s, d, _ in cfg.hypopneas:
            intervals.append((s, s + d))
        intervals.sort()
        for i in range(len(intervals) - 1):
            assert intervals[i][1] <= intervals[i + 1][0], "events must not overlap"

    def test_event_count_clipped_when_too_long(self):
        widgets = {
            "duration_s": 60.0, "fs_hz": 100.0,
            "rr_bpm": 15.0, "tv_ml": 500.0, "ie_ratio": 0.5,
            "base_pressure_cmh2o": 9.5,
            "epr_enabled": False, "epr_relief_cmh2o": 0.0,
            "n_oa": 20, "n_ca": 0, "n_ma": 0, "n_hypopnea": 0,
            "n_snore": 0, "n_cough": 0,
            "oa_mean_dur_s": 30.0, "ca_mean_dur_s": 15.0,
            "hypopnea_mean_dur_s": 20.0,
            "hr_bpm": 70.0, "cardiogenic_amplitude_cmh2o": 0.25,
            "measurement_noise_std_cmh2o": 0.05,
            "power_line_50hz_enabled": False,
            "unintentional_leak_lpm": 0.0,
        }
        cfg = ui_state.build_scenario_config(widgets, seed=0)
        assert len(cfg.obstructive_apneas) < 20
