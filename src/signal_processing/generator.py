"""Stage 0 — Synthetic respiratory signal generator.

Generates realistic CPAP pressure / flow waveforms with controllable
ground truth events for in-vitro PoC and unit testing.

Output convention
-----------------
* Pressure : cmH2O (gauge), 100 Hz, float64
* Flow    : L/min (positive = inspiration), 100 Hz, float64
* RPM     : revolutions per minute, 100 Hz, float64

The generator returns both the synthesized signals AND a `GroundTruth`
object listing every inserted event so detector accuracy can be measured.

This is the Stage 0 input for tests/demo of Stage 1~7 pipeline.

References — typical clinical values:
* Normal RR: 12~20 breaths/min (Berry RB 2012)
* Normal TV: 400~600 mL adult (Tobin 1988)
* Cardiogenic oscillation: 1~3 Hz, ~0.3 cmH2O peak (Ayappa 1999)
* CPAP intentional leak: 20~40 L/min (mask-dependent)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

EventType = Literal["normal", "OA", "CA", "MA", "hypopnea", "snore", "leak", "csr", "cough"]


@dataclass
class GroundTruthEvent:
    """A single inserted event with its true label, used for accuracy scoring."""

    type: EventType
    start_s: float
    end_s: float
    metadata: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class GroundTruth:
    """Complete ground truth for a synthesized session."""

    events: list[GroundTruthEvent] = field(default_factory=list)
    rr_bpm: float = 15.0
    tv_ml: float = 500.0
    intentional_leak_lpm: float = 24.0

    def of_type(self, t: EventType) -> list[GroundTruthEvent]:
        return [e for e in self.events if e.type == t]


# ---------------------------------------------------------------------------
# Core waveform builders
# ---------------------------------------------------------------------------


def _breath_flow(t: np.ndarray, rr_bpm: float, tv_ml: float,
                 ie_ratio: float = 1.0) -> np.ndarray:
    """Asymmetric respiratory flow waveform.

    ie_ratio = insp_time / exp_time. 1.0 = symmetric (pure sine).
    TV (inspiratory tidal volume) is preserved regardless of ie_ratio.
    """
    f_breath = rr_bpm / 60.0
    period = 1.0 / f_breath

    # Fast path: pure sine preserves bit-identical backward compatibility.
    if ie_ratio == 1.0:
        omega = 2 * np.pi * f_breath
        peak_flow_lpm = (tv_ml / 1000.0) * omega * 60.0 / 2.0
        return peak_flow_lpm * np.sin(omega * t)

    insp_dur = float(np.clip(
        period * ie_ratio / (1.0 + ie_ratio),
        0.1 * period, 0.9 * period,
    ))
    exp_dur = period - insp_dur

    phase = np.mod(t, period)
    flow = np.zeros_like(t)
    insp_mask = phase < insp_dur
    exp_mask = ~insp_mask
    flow[insp_mask] = np.sin(np.pi * phase[insp_mask] / insp_dur)
    flow[exp_mask] = -np.sin(np.pi * (phase[exp_mask] - insp_dur) / exp_dur)

    # Scale peak so ∫(insp flow) dt = TV
    insp_integral_unit = (2.0 / np.pi) * insp_dur  # area under unit half-sine
    peak_flow_lpm = (tv_ml / 1000.0) / insp_integral_unit * 60.0
    return peak_flow_lpm * flow


def _flow_to_pressure(flow_lpm: np.ndarray, base_pressure_cmh2o: float = 9.5) -> np.ndarray:
    """Approximate gauge pressure from flow.

    Real CPAP holds a base pressure (4~20 cmH2O) and modulates slightly
    with inspiratory effort. Here we use a small inspiratory dip to mimic
    real waveforms (~±0.3 cmH2O for the body of the breath).
    """
    # Inspiration causes a small pressure dip (negative correlation with flow)
    pressure_modulation = -0.03 * flow_lpm
    return base_pressure_cmh2o + pressure_modulation


# ---------------------------------------------------------------------------
# Event injection helpers
# ---------------------------------------------------------------------------


def _suppress(
    signal: np.ndarray,
    fs: float,
    start_s: float,
    end_s: float,
    factor: float,
    fade_s: float = 0.5,
) -> np.ndarray:
    """Multiply `signal` between [start_s, end_s] by `factor` with smooth fade."""
    out = signal.copy()
    n = len(signal)
    s = max(int(start_s * fs), 0)
    e = min(int(end_s * fs), n)
    if e <= s:
        return out
    fade_n = max(int(fade_s * fs), 1)
    profile = np.ones(e - s)
    if fade_n * 2 < (e - s):
        ramp = np.linspace(1.0, factor, fade_n)
        profile[:fade_n] = ramp
        profile[-fade_n:] = ramp[::-1]
        profile[fade_n:-fade_n] = factor
    else:
        profile[:] = factor
    out[s:e] *= profile
    return out


# ---------------------------------------------------------------------------
# Public synthesis API
# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """High-level scenario knobs for `synthesize_session`."""

    duration_s: float = 300.0  # 5 minutes default
    fs_hz: float = 100.0
    rr_bpm: float = 15.0
    tv_ml: float = 500.0
    base_pressure_cmh2o: float = 9.5
    intentional_leak_lpm: float = 24.0
    blower_rpm_baseline: float = 18000.0
    ie_ratio: float = 1.0  # insp_time / exp_time; 1.0 = symmetric (1:1)

    # Event schedule (start_s, duration_s)
    obstructive_apneas: list[tuple[float, float]] = field(default_factory=list)
    central_apneas: list[tuple[float, float]] = field(default_factory=list)
    hypopneas: list[tuple[float, float, float]] = field(default_factory=list)
    # ^ (start, duration, severity 0~1 — fraction of normal flow remaining)
    snore_episodes: list[tuple[float, float]] = field(default_factory=list)
    mixed_apneas: list[tuple[float, float, float]] = field(default_factory=list)
    # ^ (start_s, total_duration_s, central_fraction 0~1)
    unintentional_leak_lpm: float = 0.0
    unintentional_leak_profile: str = "constant"  # "constant" | "ramp" | "burst"
    cough_events: list[tuple[float, float]] = field(default_factory=list)
    # ^ (start_s, peak_amplitude_cmh2o)
    power_line_50hz_amplitude_cmh2o: float = 0.0
    epr_enabled: bool = False
    epr_relief_cmh2o: float = 0.0

    # Always-on physiological
    cardiogenic_amplitude_cmh2o: float = 0.25
    heart_rate_bpm: float = 70.0
    measurement_noise_std_cmh2o: float = 0.05


def synthesize_session(cfg: ScenarioConfig, seed: int | None = 42) -> tuple[
    dict[str, np.ndarray], GroundTruth
]:
    """Synthesize a full multi-event session.

    Returns
    -------
    signals : dict
        {
          't_s'           : time vector (s),
          'pressure_cmh2o': gauge pressure 100 Hz,
          'flow_lpm'      : true flow 100 Hz (ground truth — for validation only),
          'blower_rpm'    : blower RPM 100 Hz (for Phase 0 estimation),
        }
    ground_truth : GroundTruth
        All inserted events for later accuracy scoring.
    """
    rng = np.random.default_rng(seed)
    n = int(cfg.duration_s * cfg.fs_hz)
    t = np.arange(n) / cfg.fs_hz

    # --- 1. Base normal breathing ---
    flow_patient = _breath_flow(t, cfg.rr_bpm, cfg.tv_ml, cfg.ie_ratio)
    pressure = _flow_to_pressure(flow_patient, cfg.base_pressure_cmh2o)
    # EPR: drop pressure during expiration (flow < 0)
    if cfg.epr_enabled and cfg.epr_relief_cmh2o > 0.0:
        epr_drop = np.where(flow_patient < 0, cfg.epr_relief_cmh2o, 0.0)
        win = max(int(0.1 * cfg.fs_hz), 1)
        kernel = np.ones(win) / win
        epr_drop = np.convolve(epr_drop, kernel, mode="same")
        pressure = pressure - epr_drop

    gt = GroundTruth(rr_bpm=cfg.rr_bpm, tv_ml=cfg.tv_ml,
                     intentional_leak_lpm=cfg.intentional_leak_lpm)

    # --- 2. Inject obstructive apneas (kill flow + REMOVE cardiogenic) ---
    cardiogenic_mask = np.ones(n)  # 1 = cardiogenic present (open airway)
    for start_s, dur in cfg.obstructive_apneas:
        end_s = start_s + dur
        # Flow drops to ~5% of normal
        flow_patient = _suppress(flow_patient, cfg.fs_hz, start_s, end_s, factor=0.05)
        # Pressure modulation also drops (no breathing motion)
        pressure_mod = pressure - cfg.base_pressure_cmh2o
        pressure_mod = _suppress(pressure_mod, cfg.fs_hz, start_s, end_s, factor=0.05)
        pressure = cfg.base_pressure_cmh2o + pressure_mod
        # Block cardiogenic transmission (closed airway)
        cardiogenic_mask = _suppress(
            cardiogenic_mask, cfg.fs_hz, start_s, end_s, factor=0.05
        )
        gt.events.append(GroundTruthEvent("OA", start_s, end_s))

    # --- 3. Inject central apneas (kill flow but KEEP cardiogenic) ---
    for start_s, dur in cfg.central_apneas:
        end_s = start_s + dur
        flow_patient = _suppress(flow_patient, cfg.fs_hz, start_s, end_s, factor=0.05)
        pressure_mod = pressure - cfg.base_pressure_cmh2o
        pressure_mod = _suppress(pressure_mod, cfg.fs_hz, start_s, end_s, factor=0.05)
        pressure = cfg.base_pressure_cmh2o + pressure_mod
        # Cardiogenic mask stays 1 (open airway) → cardiogenic transmits
        gt.events.append(GroundTruthEvent("CA", start_s, end_s))

    # --- 4. Inject hypopneas ---
    for start_s, dur, severity in cfg.hypopneas:
        end_s = start_s + dur
        flow_patient = _suppress(
            flow_patient, cfg.fs_hz, start_s, end_s, factor=severity
        )
        pressure_mod = pressure - cfg.base_pressure_cmh2o
        pressure_mod = _suppress(
            pressure_mod, cfg.fs_hz, start_s, end_s, factor=severity
        )
        pressure = cfg.base_pressure_cmh2o + pressure_mod
        gt.events.append(GroundTruthEvent(
            "hypopnea", start_s, end_s, metadata={"severity": severity}
        ))

    # --- 4b. Inject mixed apneas (central → obstructive transition) ---
    for start_s, total_dur, central_fraction in cfg.mixed_apneas:
        central_fraction = float(np.clip(central_fraction, 0.0, 1.0))
        mid_s = start_s + total_dur * central_fraction
        end_s = start_s + total_dur
        flow_patient = _suppress(flow_patient, cfg.fs_hz, start_s, end_s, factor=0.05)
        pressure_mod = pressure - cfg.base_pressure_cmh2o
        pressure_mod = _suppress(pressure_mod, cfg.fs_hz, start_s, end_s, factor=0.05)
        pressure = cfg.base_pressure_cmh2o + pressure_mod
        cardiogenic_mask = _suppress(
            cardiogenic_mask, cfg.fs_hz, mid_s, end_s, factor=0.05
        )
        gt.events.append(GroundTruthEvent(
            "MA", start_s, end_s,
            metadata={"central_fraction": central_fraction, "transition_s": mid_s},
        ))

    # --- 5. Add cardiogenic oscillation (1~3 Hz, modulated by airway state) ---
    f_heart = cfg.heart_rate_bpm / 60.0
    cardiogenic_signal = (
        cfg.cardiogenic_amplitude_cmh2o
        * np.sin(2 * np.pi * f_heart * t)
        * cardiogenic_mask
    )
    pressure = pressure + cardiogenic_signal

    # --- 6. Add snoring (60~200 Hz high-frequency content) ---
    for start_s, dur in cfg.snore_episodes:
        end_s = start_s + dur
        s = max(int(start_s * cfg.fs_hz), 0)
        e = min(int(end_s * cfg.fs_hz), n)
        # 100 Hz snore signal — frequency near sampling Nyquist limit
        snore = 0.4 * np.sin(2 * np.pi * 40.0 * t[s:e])  # ~40 Hz visible at fs=100
        snore += 0.2 * rng.standard_normal(e - s)
        pressure[s:e] += snore
        gt.events.append(GroundTruthEvent("snore", start_s, end_s))

    # --- 6b. Inject cough events (gaussian-shaped pressure spike) ---
    for start_s, peak_amp in cfg.cough_events:
        center = int(start_s * cfg.fs_hz)
        sigma_samples = int(0.1 * cfg.fs_hz)  # 100 ms half-width
        win_n = sigma_samples * 6
        idx = np.arange(-win_n // 2, win_n // 2)
        kernel = peak_amp * np.exp(-(idx ** 2) / (2 * sigma_samples ** 2))
        s = max(center + idx[0], 0)
        e = min(center + idx[-1] + 1, n)
        ks = s - (center + idx[0])
        ke = ks + (e - s)
        pressure[s:e] += kernel[ks:ke]
        gt.events.append(GroundTruthEvent(
            "cough", start_s, start_s + win_n / cfg.fs_hz,
            metadata={"peak_amplitude_cmh2o": float(peak_amp)},
        ))

    # --- 6c. Inject 50 Hz power-line interference ---
    if cfg.power_line_50hz_amplitude_cmh2o > 0.0 and 50.0 < cfg.fs_hz / 2.0:
        pressure = pressure + cfg.power_line_50hz_amplitude_cmh2o * np.sin(
            2 * np.pi * 50.0 * t
        )

    # --- 7. Add measurement noise ---
    noise = rng.normal(0.0, cfg.measurement_noise_std_cmh2o, n)
    pressure = pressure + noise

    # --- 8. Total flow including intentional + unintentional leak ---
    leak_profile = np.zeros(n)
    if cfg.unintentional_leak_lpm > 0.0:
        if cfg.unintentional_leak_profile == "constant":
            leak_profile += cfg.unintentional_leak_lpm
        elif cfg.unintentional_leak_profile == "ramp":
            leak_profile += np.linspace(0.0, cfg.unintentional_leak_lpm, n)
        elif cfg.unintentional_leak_profile == "burst":
            burst_start = int(0.4 * n)
            burst_end = int(0.6 * n)
            leak_profile[burst_start:burst_end] += cfg.unintentional_leak_lpm
    flow_total = flow_patient + cfg.intentional_leak_lpm + leak_profile

    # --- 9. Blower RPM from closed-loop demand ---
    # Real CPAP: blower delivers (patient_flow + intentional_leak) at constant
    # pressure setpoint by modulating RPM. The blower lookup table is
    #   Q ≈ rpm * (a + b * P)   where a=0.00135, b=-4.5e-6 (see flow_estimator)
    # Inverting: rpm = Q / (a + b * P)
    a_coef = 0.00135
    b_coef = -4.5e-6
    denom = np.clip(a_coef + b_coef * pressure, 1e-4, None)
    blower_rpm = flow_total / denom
    blower_rpm += rng.normal(0.0, 50.0, n)  # RPM jitter (sensor noise)

    signals = {
        "t_s": t,
        "pressure_cmh2o": pressure,
        "flow_lpm": flow_total,            # ground truth — never use in detector
        "flow_patient_lpm": flow_patient,  # patient component only (no leak)
        "blower_rpm": blower_rpm,
    }
    return signals, gt


def default_demo_scenario() -> ScenarioConfig:
    """A representative 5-minute scenario covering all event types.

    Schedule (start_s, end_s):
      00:00 ~ 01:00  Normal breathing
      01:00 ~ 01:18  Obstructive Apnea (18 s, OA)
      01:18 ~ 02:00  Recovery breathing
      02:00 ~ 02:15  Central Apnea (15 s, CA)
      02:15 ~ 02:50  Recovery
      02:50 ~ 03:10  Hypopnea (20 s, severity=0.4)
      03:10 ~ 03:40  Snore episode
      03:40 ~ 04:00  Recovery
      04:00 ~ 04:25  Obstructive Apnea (25 s, deeper)
      04:25 ~ 05:00  Recovery
    """
    return ScenarioConfig(
        duration_s=300.0,
        fs_hz=100.0,
        rr_bpm=15.0,
        tv_ml=500.0,
        obstructive_apneas=[(60.0, 18.0), (240.0, 25.0)],
        central_apneas=[(120.0, 15.0)],
        hypopneas=[(170.0, 20.0, 0.4)],
        snore_episodes=[(190.0, 30.0)],
    )
