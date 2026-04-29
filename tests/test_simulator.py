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


from simulator import randomizer  # noqa: E402


class TestRandomizer:
    def test_randomize_within_clinical_ranges(self):
        widgets = randomizer.randomize_widgets(seed=42)
        assert 60.0 <= widgets["duration_s"] <= 1800.0
        assert 8.0 <= widgets["rr_bpm"] <= 30.0
        assert 200.0 <= widgets["tv_ml"] <= 800.0
        assert 4.0 <= widgets["base_pressure_cmh2o"] <= 20.0
        assert 40.0 <= widgets["hr_bpm"] <= 100.0
        assert 0 <= widgets["n_oa"] <= 20
        assert 0 <= widgets["n_hypopnea"] <= 20

    def test_pinned_seed_produces_identical_output(self):
        a = randomizer.randomize_widgets(seed=123)
        b = randomizer.randomize_widgets(seed=123)
        assert a == b


import io
import json
import zipfile

from simulator import exporter  # noqa: E402


class TestExporter:
    def _build_session(self, seed=42):
        cfg = generator.default_demo_scenario()
        signals, gt = generator.synthesize_session(cfg, seed=seed)
        return cfg, signals, gt

    def test_zip_contains_four_files(self):
        cfg, signals, gt = self._build_session()
        zip_bytes = exporter.build_zip(signals, gt, cfg, seed=42)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            names = set(z.namelist())
        assert names == {"signal.csv", "ground_truth.csv", "metadata.json", "README.txt"}

    def test_signal_csv_row_count_matches_duration_fs(self):
        cfg, signals, gt = self._build_session()
        zip_bytes = exporter.build_zip(signals, gt, cfg, seed=42)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            with z.open("signal.csv") as f:
                lines = f.read().decode().strip().split("\n")
        n_data = len(lines) - 1
        assert n_data == int(cfg.duration_s * cfg.fs_hz)

    def test_ground_truth_csv_count_matches(self):
        cfg, signals, gt = self._build_session()
        zip_bytes = exporter.build_zip(signals, gt, cfg, seed=42)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            with z.open("ground_truth.csv") as f:
                lines = f.read().decode().strip().split("\n")
        n_events = len(lines) - 1
        assert n_events == len(gt.events)

    def test_metadata_json_schema_v1(self):
        cfg, signals, gt = self._build_session()
        zip_bytes = exporter.build_zip(signals, gt, cfg, seed=42)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            meta = json.loads(z.read("metadata.json").decode())
        assert meta["schema_version"] == "1.0"
        assert meta["seed"] == 42
        assert meta["fs_hz"] == cfg.fs_hz
        assert "scenario" in meta
        assert "event_summary" in meta

    def test_csv_readable_by_pandas(self):
        import pandas as pd
        cfg, signals, gt = self._build_session()
        zip_bytes = exporter.build_zip(signals, gt, cfg, seed=42)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            with z.open("signal.csv") as f:
                df = pd.read_csv(f)
        assert "pressure_cmh2o" in df.columns
        assert "time_s" in df.columns
        assert len(df) == int(cfg.duration_s * cfg.fs_hz)
