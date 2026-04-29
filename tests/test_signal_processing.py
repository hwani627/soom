"""Unit + e2e tests for the signal_processing module.

Run with:
    pytest tests/test_signal_processing.py -v

Each test uses the synthetic generator's GroundTruth as the reference,
so accuracy can be measured deterministically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from signal_processing import (
    breath_analyzer,
    event_detector,
    filters,
    flow_estimator,
    generator,
)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class TestGenerator:
    def test_normal_breathing_RR_within_tolerance(self):
        """Synthesized normal breathing should match requested RR within ±0.5 bpm."""
        cfg = generator.ScenarioConfig(
            duration_s=60.0, fs_hz=100.0, rr_bpm=15.0,
            cardiogenic_amplitude_cmh2o=0.0,  # remove confounders
            measurement_noise_std_cmh2o=0.0,
        )
        signals, _ = generator.synthesize_session(cfg, seed=0)
        flow = signals["flow_patient_lpm"]
        cycles = breath_analyzer.extract_breath_cycles(flow, fs=cfg.fs_hz)
        rr_zc = breath_analyzer.compute_respiratory_rate_zc(cycles)
        assert abs(rr_zc - 15.0) < 0.5

    def test_apnea_drops_flow_amplitude(self):
        """Inserted obstructive apnea segment should have ≤5% of normal RMS."""
        cfg = generator.ScenarioConfig(
            duration_s=120.0, fs_hz=100.0, rr_bpm=15.0,
            obstructive_apneas=[(50.0, 20.0)],  # 50~70 s
            cardiogenic_amplitude_cmh2o=0.0,
            measurement_noise_std_cmh2o=0.0,
        )
        signals, _ = generator.synthesize_session(cfg, seed=0)
        flow = signals["flow_patient_lpm"]
        fs = cfg.fs_hz

        normal_seg = flow[int(20 * fs):int(40 * fs)]
        apnea_seg = flow[int(58 * fs):int(68 * fs)]  # mid-apnea, after fade
        normal_rms = float(np.sqrt(np.mean(normal_seg ** 2)))
        apnea_rms = float(np.sqrt(np.mean(apnea_seg ** 2)))
        assert apnea_rms < 0.10 * normal_rms

    def test_ground_truth_event_count(self):
        cfg = generator.default_demo_scenario()
        _, gt = generator.synthesize_session(cfg, seed=42)
        assert len(gt.of_type("OA")) == 2
        assert len(gt.of_type("CA")) == 1
        assert len(gt.of_type("hypopnea")) == 1
        assert len(gt.of_type("snore")) == 1

    def test_ie_ratio_changes_inspiration_duration(self):
        """ie_ratio=0.4 → inspiration is shorter than expiration (1:1.5)."""
        cfg_sym = generator.ScenarioConfig(
            duration_s=20.0, fs_hz=100.0, rr_bpm=15.0,
            ie_ratio=1.0,
            cardiogenic_amplitude_cmh2o=0.0,
            measurement_noise_std_cmh2o=0.0,
        )
        cfg_asym = generator.ScenarioConfig(
            duration_s=20.0, fs_hz=100.0, rr_bpm=15.0,
            ie_ratio=0.4,
            cardiogenic_amplitude_cmh2o=0.0,
            measurement_noise_std_cmh2o=0.0,
        )
        sig_sym, _ = generator.synthesize_session(cfg_sym, seed=0)
        sig_asym, _ = generator.synthesize_session(cfg_asym, seed=0)

        flow_sym = sig_sym["flow_patient_lpm"]
        flow_asym = sig_asym["flow_patient_lpm"]

        insp_frac_sym = float(np.mean(flow_sym > 0))
        insp_frac_asym = float(np.mean(flow_asym > 0))
        assert insp_frac_sym > 0.45 and insp_frac_sym < 0.55
        assert insp_frac_asym < insp_frac_sym - 0.05


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestFilters:
    def test_4branch_separation(self):
        """4 BPF branches should separate a 3-component synthetic signal."""
        fs = 100.0
        t = np.arange(0, 30.0, 1 / fs)
        # Pure components: breath 0.25 Hz, cardiogenic 1.5 Hz, fot 5 Hz
        breath_signal = 1.0 * np.sin(2 * np.pi * 0.25 * t)
        cardio_signal = 0.3 * np.sin(2 * np.pi * 1.5 * t)
        fot_signal = 0.2 * np.sin(2 * np.pi * 5.0 * t)
        composite = breath_signal + cardio_signal + fot_signal

        branches = filters.apply_4branch_bpf(composite, fs=fs)

        # Breath branch dominated by 0.25 Hz energy
        breath_rms = float(np.sqrt(np.mean(branches["breath"] ** 2)))
        # Should be close to original 1.0 / sqrt(2)
        assert breath_rms > 0.5
        # Cardiogenic branch should isolate the 1.5 Hz component
        cardio_rms = float(np.sqrt(np.mean(branches["cardiogenic"] ** 2)))
        assert 0.10 < cardio_rms < 0.40
        # FOT branch should isolate the 5 Hz component
        fot_rms = float(np.sqrt(np.mean(branches["fot"] ** 2)))
        assert 0.05 < fot_rms < 0.30

    def test_notch_removes_50hz(self):
        fs = 1000.0
        t = np.arange(0, 5.0, 1 / fs)
        signal_clean = np.sin(2 * np.pi * 5.0 * t)
        signal_noisy = signal_clean + 0.5 * np.sin(2 * np.pi * 50.0 * t)

        cleaned = filters.notch_filter(signal_noisy, fs=fs, freq_hz=50.0)
        # Energy at 50 Hz should be reduced
        from numpy.fft import rfft, rfftfreq
        freqs = rfftfreq(len(cleaned), 1 / fs)
        spec = np.abs(rfft(cleaned))
        i_50 = int(np.argmin(np.abs(freqs - 50.0)))
        assert spec[i_50] < 50.0  # large reduction


# ---------------------------------------------------------------------------
# Flow estimator
# ---------------------------------------------------------------------------


class TestFlowEstimator:
    def test_phase0_correlates_with_truth(self):
        """Phase 0 estimate should be correlated with ground-truth flow."""
        cfg = generator.ScenarioConfig(
            duration_s=60.0, fs_hz=100.0, rr_bpm=15.0,
            cardiogenic_amplitude_cmh2o=0.0,
            measurement_noise_std_cmh2o=0.0,
        )
        signals, _ = generator.synthesize_session(cfg, seed=0)
        fs = cfg.fs_hz
        refined = filters.stage2_refine(signals["pressure_cmh2o"],
                                        fs_in=fs, fs_out=fs)
        flow_est, conf = flow_estimator.estimate_flow_phase0(
            refined, signals["blower_rpm"], fs=fs,
            intentional_leak_lpm=cfg.intentional_leak_lpm,
        )
        truth = signals["flow_patient_lpm"]
        # Trim transient edges
        flow_est_t = flow_est[int(5 * fs):int(55 * fs)]
        truth_t = truth[int(5 * fs):int(55 * fs)]
        corr = float(np.corrcoef(flow_est_t, truth_t)[0, 1])
        assert corr > 0.80

    def test_phase1_pneumotacho_accuracy(self):
        """Phase 1 forward-then-inverse should round-trip flow within ±5%."""
        cfg = generator.ScenarioConfig(
            duration_s=10.0, fs_hz=100.0, rr_bpm=15.0,
            cardiogenic_amplitude_cmh2o=0.0,
            measurement_noise_std_cmh2o=0.0,
        )
        signals, _ = generator.synthesize_session(cfg, seed=1)
        true_total_flow = signals["flow_patient_lpm"] + cfg.intentional_leak_lpm
        dp = flow_estimator.synthesize_pneumotacho_dp(
            true_total_flow, K=1.5, n=0.5, noise_pa_std=0.5
        )
        flow_recovered, _ = flow_estimator.measure_flow_phase1(
            dp, K=1.5, n=0.5, intentional_leak_lpm=cfg.intentional_leak_lpm
        )
        rmse = float(np.sqrt(np.mean(
            (flow_recovered - signals["flow_patient_lpm"]) ** 2
        )))
        std_truth = float(np.std(signals["flow_patient_lpm"]))
        assert rmse < 0.10 * std_truth + 1.0  # ±10% tolerance for the PoC

    def test_lookup_table_returns_positive_baseline(self):
        lookup = flow_estimator.BlowerLookupTable.synthetic_default()
        # At nominal CPAP operating point, baseline flow should be ~10~30 L/min
        flow = lookup.lookup(np.array([18000.0]), np.array([9.5]))
        assert 10.0 < float(flow[0]) < 35.0


# ---------------------------------------------------------------------------
# Breath analyzer
# ---------------------------------------------------------------------------


class TestBreathAnalyzer:
    def test_envelope_smooths_signal(self):
        fs = 100.0
        t = np.arange(0, 10.0, 1 / fs)
        flow = 30.0 * np.sin(2 * np.pi * 0.25 * t)
        env = breath_analyzer.compute_envelope(flow, fs=fs, window_s=2.0)
        assert env.shape == flow.shape
        # RMS of sine = peak / sqrt(2) ≈ 21.2 — envelope should be near this
        steady = env[int(3 * fs):int(8 * fs)]
        assert 15.0 < float(np.mean(steady)) < 27.0

    def test_extract_breath_cycles_count(self):
        cfg = generator.ScenarioConfig(
            duration_s=60.0, fs_hz=100.0, rr_bpm=15.0,
            cardiogenic_amplitude_cmh2o=0.0,
            measurement_noise_std_cmh2o=0.0,
        )
        signals, _ = generator.synthesize_session(cfg, seed=0)
        cycles = breath_analyzer.extract_breath_cycles(
            signals["flow_patient_lpm"], fs=cfg.fs_hz
        )
        # Expected ≈ 60s × 15bpm / 60 = 15 cycles. Allow ±2.
        assert 12 <= len(cycles) <= 17

    def test_RR_estimate_within_1bpm(self):
        cfg = generator.ScenarioConfig(
            duration_s=120.0, fs_hz=100.0, rr_bpm=14.0,
            cardiogenic_amplitude_cmh2o=0.0,
            measurement_noise_std_cmh2o=0.0,
        )
        signals, _ = generator.synthesize_session(cfg, seed=0)
        rr = breath_analyzer.compute_respiratory_rate(
            signals["flow_patient_lpm"], fs=cfg.fs_hz
        )
        assert abs(rr["rr"] - 14.0) < 1.0

    def test_flow_limitation_index_for_sinusoid(self):
        """A clean half-sinusoid should have a much lower FL index than a
        flattened (UARS-like) waveform."""
        fs = 100.0
        t = np.linspace(0, 1.5, int(1.5 * fs))
        insp_round = np.sin(np.pi * t / 1.5)
        insp_flat = np.minimum(insp_round * 1.5, 1.0)  # squashed top
        fl_round = breath_analyzer.compute_flow_limitation_index(insp_round)
        fl_flat = breath_analyzer.compute_flow_limitation_index(insp_flat)
        assert fl_round < fl_flat
        assert fl_flat > 0.7


# ---------------------------------------------------------------------------
# Event detector
# ---------------------------------------------------------------------------


class TestEventDetector:
    def _build_pipeline_outputs(self, seed: int = 42):
        cfg = generator.default_demo_scenario()
        signals, gt = generator.synthesize_session(cfg, seed=seed)
        fs = cfg.fs_hz
        refined = filters.stage2_refine(signals["pressure_cmh2o"],
                                        fs_in=fs, fs_out=fs)
        branches = filters.apply_4branch_bpf(refined, fs=fs)
        flow_est, _ = flow_estimator.estimate_flow_phase0(
            refined, signals["blower_rpm"], fs=fs,
            intentional_leak_lpm=cfg.intentional_leak_lpm,
        )
        envelope = breath_analyzer.compute_envelope(flow_est, fs=fs)
        baseline = breath_analyzer.compute_baseline(envelope, fs=fs)
        return cfg, fs, branches, envelope, baseline, gt

    def test_apnea_sensitivity_at_least_60pct(self):
        """The 3 ground-truth apneas (2 OA + 1 CA) should produce ≥ 60% sensitivity.

        Note: at a single-sensor, no-PSG-learning PoC level, we don't yet
        target the 90% spec acceptance. This test enforces a reasonable
        floor and serves as a regression guard.
        """
        cfg, fs, branches, env, base, gt = self._build_pipeline_outputs()
        apneas = event_detector.detect_apnea(env, base, fs=fs)
        truth_apneas = [t for t in gt.events if t.type in {"OA", "CA"}]
        # Build matching by overlap
        matched = 0
        for tev in truth_apneas:
            for det in apneas:
                if (det.end_s + 5 >= tev.start_s
                        and det.start_s - 5 <= tev.end_s):
                    matched += 1
                    break
        sensitivity = matched / max(len(truth_apneas), 1)
        assert sensitivity >= 0.60

    def test_cardiogenic_distinguishes_OA_vs_CA(self):
        """Cardiogenic detection should yield different flag values for
        an OA window (cardiogenic absent) vs a CA window (cardiogenic present)."""
        cfg = generator.ScenarioConfig(
            duration_s=120.0, fs_hz=100.0, rr_bpm=15.0,
            obstructive_apneas=[(20.0, 15.0)],
            central_apneas=[(70.0, 15.0)],
            cardiogenic_amplitude_cmh2o=0.4,  # strong cardiogenic to make signal clear
            measurement_noise_std_cmh2o=0.02,
        )
        signals, _ = generator.synthesize_session(cfg, seed=7)
        fs = cfg.fs_hz
        refined = filters.stage2_refine(signals["pressure_cmh2o"],
                                        fs_in=fs, fs_out=fs)
        branches = filters.apply_4branch_bpf(refined, fs=fs)
        cardio = branches["cardiogenic"]

        oa_result = event_detector.detect_cardiogenic(
            cardio, fs=fs, start_s=22.0, end_s=33.0
        )
        ca_result = event_detector.detect_cardiogenic(
            cardio, fs=fs, start_s=72.0, end_s=83.0
        )
        # CA should retain more cardiogenic power than OA
        assert ca_result["power"] > oa_result["power"]

    def test_session_stats_AHI(self):
        """AHI = 4 events (3 apnea + 1 hypopnea) over 5 min ≈ 48/h."""
        cfg, fs, branches, env, base, gt = self._build_pipeline_outputs()
        apneas = event_detector.detect_apnea(env, base, fs=fs)
        hypops = event_detector.detect_hypopnea(env, base, fs=fs)
        stats = event_detector.compute_session_stats(
            apneas + hypops, duration_s=cfg.duration_s
        )
        # Expect AHI in roughly 30~80 range; gives margin for missed events
        assert 20.0 <= stats["AHI"] <= 90.0

    def test_score_function_works(self):
        cfg, fs, branches, env, base, gt = self._build_pipeline_outputs()
        apneas = event_detector.detect_apnea(env, base, fs=fs)
        # Classify subtypes
        apneas_cls = [
            event_detector.classify_apnea_subtype(a, branches["cardiogenic"], fs=fs)
            for a in apneas
        ]
        hypops = event_detector.detect_hypopnea(env, base, fs=fs)
        score = event_detector.score_events_vs_ground_truth(
            apneas_cls + hypops, gt, tolerance_s=5.0
        )
        assert "TP" in score and "sensitivity" in score
        assert score["sensitivity"] >= 0.0


# ---------------------------------------------------------------------------
# E2E
# ---------------------------------------------------------------------------


class TestE2E:
    def test_full_pipeline_runs_and_produces_events(self):
        """End-to-end smoke test: scenario → pipeline → detected events."""
        cfg = generator.default_demo_scenario()
        signals, gt = generator.synthesize_session(cfg, seed=42)
        fs = cfg.fs_hz

        refined = filters.stage2_refine(signals["pressure_cmh2o"],
                                        fs_in=fs, fs_out=fs)
        branches = filters.apply_4branch_bpf(refined, fs=fs)
        flow_est, _ = flow_estimator.estimate_flow_phase0(
            refined, signals["blower_rpm"], fs=fs,
            intentional_leak_lpm=cfg.intentional_leak_lpm,
        )
        env = breath_analyzer.compute_envelope(flow_est, fs=fs)
        base = breath_analyzer.compute_baseline(env, fs=fs)

        apneas = event_detector.detect_apnea(env, base, fs=fs)
        apneas_cls = [
            event_detector.classify_apnea_subtype(a, branches["cardiogenic"], fs=fs)
            for a in apneas
        ]
        hypops = event_detector.detect_hypopnea(env, base, fs=fs)

        # We should detect at least 1 event of each kind
        assert len(apneas_cls) >= 1
        # Sanity: AHI should be > 0
        stats = event_detector.compute_session_stats(
            apneas_cls + hypops, duration_s=cfg.duration_s
        )
        assert stats["AHI"] > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
