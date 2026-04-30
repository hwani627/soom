"""Randomize slider values within clinically plausible ranges."""
from __future__ import annotations

import numpy as np


def randomize_widgets(seed: int) -> dict:
    """Sample slider values from clinical normal ranges.

    Returns a dict shaped like the UI widget state, ready to feed into
    ui_state.build_scenario_config().
    """
    rng = np.random.default_rng(seed)
    return {
        "duration_s": float(rng.choice([180.0, 300.0, 600.0, 900.0])),
        "fs_hz": 100.0,
        "rr_bpm": float(rng.uniform(12.0, 20.0)),
        "tv_ml": float(rng.uniform(380.0, 620.0)),
        "ie_ratio": float(rng.uniform(0.4, 0.7)),
        "base_pressure_cmh2o": float(rng.uniform(6.0, 14.0)),
        "epr_enabled": bool(rng.random() < 0.3),
        "epr_relief_cmh2o": float(rng.uniform(1.0, 2.5)),
        "n_oa": int(rng.integers(0, 6)),
        "n_ca": int(rng.integers(0, 3)),
        "n_ma": int(rng.integers(0, 2)),
        "n_hypopnea": int(rng.integers(0, 5)),
        "n_snore": int(rng.integers(0, 3)),
        "n_cough": int(rng.integers(0, 3)),
        "oa_mean_dur_s": float(rng.uniform(14.0, 30.0)),
        "ca_mean_dur_s": float(rng.uniform(12.0, 25.0)),
        "hypopnea_mean_dur_s": float(rng.uniform(12.0, 30.0)),
        "hr_bpm": float(rng.uniform(55.0, 85.0)),
        "cardiogenic_amplitude_cmh2o": float(rng.uniform(0.15, 0.40)),
        "measurement_noise_std_cmh2o": float(rng.uniform(0.02, 0.10)),
        "power_line_50hz_enabled": bool(rng.random() < 0.4),
        "unintentional_leak_lpm": float(rng.choice([0.0, 0.0, 0.0, 10.0, 20.0])),
        # intentional_leak_lpm varies with mask type (20~40 L/min); randomize within range
        "intentional_leak_lpm": float(rng.uniform(20.0, 40.0)),
    }
