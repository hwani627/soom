"""Convert Streamlit widget state into a fully-formed ScenarioConfig."""
from __future__ import annotations

import numpy as np

from signal_processing.generator import ScenarioConfig


def _allocate_event_intervals(
    n_events: int,
    mean_duration_s: float,
    duration_s: float,
    occupied: list[tuple[float, float]],
    rng: np.random.Generator,
    margin_s: float = 5.0,
    max_attempts_per_event: int = 50,
) -> list[tuple[float, float]]:
    """Place n_events of approximate mean duration without overlapping occupied."""
    placed: list[tuple[float, float]] = []
    if duration_s < margin_s * 2:
        return placed
    for _ in range(n_events):
        dur = float(np.clip(
            rng.normal(mean_duration_s, mean_duration_s * 0.15),
            max(mean_duration_s * 0.5, 5.0),
            mean_duration_s * 1.5,
        ))
        for _attempt in range(max_attempts_per_event):
            start = float(rng.uniform(margin_s, duration_s - dur - margin_s))
            end = start + dur
            collision = any(
                not (end + margin_s < s or start - margin_s > e)
                for s, e in (*occupied, *placed)
            )
            if not collision:
                placed.append((start, end))
                break
    return placed


def build_scenario_config(widgets: dict, seed: int) -> ScenarioConfig:
    """Build a ScenarioConfig from a dict of slider/widget values."""
    rng = np.random.default_rng(seed)
    duration_s = float(widgets["duration_s"])

    occupied: list[tuple[float, float]] = []

    def alloc(n: int, mean_dur: float) -> list[tuple[float, float]]:
        intervals = _allocate_event_intervals(n, mean_dur, duration_s, occupied, rng)
        for s, e in intervals:
            occupied.append((s, e))
        return [(s, e - s) for s, e in intervals]

    oa = alloc(int(widgets["n_oa"]), float(widgets["oa_mean_dur_s"]))
    ca = alloc(int(widgets["n_ca"]), float(widgets["ca_mean_dur_s"]))
    ma_intervals = alloc(int(widgets.get("n_ma", 0)), float(widgets["oa_mean_dur_s"]))
    hyps_raw = alloc(int(widgets["n_hypopnea"]), float(widgets["hypopnea_mean_dur_s"]))
    hyps = [(s, d, float(rng.uniform(0.3, 0.6))) for s, d in hyps_raw]
    snores = alloc(int(widgets["n_snore"]), 30.0)
    cough_intervals = alloc(int(widgets.get("n_cough", 0)), 0.5)
    coughs = [(s, float(rng.uniform(2.0, 4.0))) for s, _ in cough_intervals]
    mas = [(s, d, float(rng.uniform(0.4, 0.7))) for s, d in ma_intervals]

    return ScenarioConfig(
        duration_s=duration_s,
        fs_hz=float(widgets["fs_hz"]),
        rr_bpm=float(widgets["rr_bpm"]),
        tv_ml=float(widgets["tv_ml"]),
        ie_ratio=float(widgets["ie_ratio"]),
        base_pressure_cmh2o=float(widgets["base_pressure_cmh2o"]),
        intentional_leak_lpm=float(widgets.get("intentional_leak_lpm", 24.0)),
        obstructive_apneas=oa,
        central_apneas=ca,
        mixed_apneas=mas,
        hypopneas=hyps,
        snore_episodes=snores,
        cough_events=coughs,
        cardiogenic_amplitude_cmh2o=float(widgets["cardiogenic_amplitude_cmh2o"]),
        heart_rate_bpm=float(widgets["hr_bpm"]),
        measurement_noise_std_cmh2o=float(widgets["measurement_noise_std_cmh2o"]),
        power_line_50hz_amplitude_cmh2o=(
            0.3 if widgets.get("power_line_50hz_enabled") else 0.0
        ),
        epr_enabled=bool(widgets.get("epr_enabled", False)),
        epr_relief_cmh2o=float(widgets.get("epr_relief_cmh2o", 0.0)),
        unintentional_leak_lpm=float(widgets.get("unintentional_leak_lpm", 0.0)),
    )
