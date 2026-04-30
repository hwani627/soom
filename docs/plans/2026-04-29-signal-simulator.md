# Signal Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `generator.py`의 `ScenarioConfig`를 인터랙티브 Streamlit 페이지로 감싸, 슬라이더·🎲Randomize로 양압기 압력 raw 신호 시나리오를 만들고 외부 알고리즘 검증용 CSV+JSON ZIP으로 내보내기.

**Architecture:** Streamlit multi-page (`pages/2_📈_Signal_Simulator.py`)에서 사이드바 위젯 → `simulator/ui_state` → `ScenarioConfig` → `generator.synthesize_session()` → `simulator/plotting`+`simulator/exporter`. UI 모듈은 generator를 모르고 generator는 UI를 모름 (단방향 의존). Generator에 Tier 2 항목(ie_ratio, MA, unintentional leak, cough, 50Hz, EPR) 추가는 모두 default 무동작 → 기존 17개 테스트 backward compatible.

**Tech Stack:** Python 3.11+, Streamlit, NumPy, SciPy, Pandas, Plotly, pytest. 신규 의존성 없음.

**Spec reference:** `docs/specs/2026-04-29-signal-simulator-design.md` (commit 88b19bc).

---

## Task 1: Generator — `ie_ratio` (비대칭 흡/호기 파형)

**Files:**
- Modify: `src/signal_processing/generator.py` (`ScenarioConfig` 데이터클래스 + `_breath_flow` 함수)
- Test: `tests/test_signal_processing.py` (TestGenerator 클래스에 추가)

- [ ] **Step 1: Write failing test**

`tests/test_signal_processing.py`의 `TestGenerator` 클래스 끝에 추가:
```python
    def test_ie_ratio_changes_inspiration_duration(self):
        """ie_ratio=0.4 → inspiration is shorter than expiration (1:1.5)."""
        cfg_sym = generator.ScenarioConfig(
            duration_s=20.0, fs_hz=100.0, rr_bpm=15.0,
            ie_ratio=1.0,  # symmetric (current sine wave)
            cardiogenic_amplitude_cmh2o=0.0,
            measurement_noise_std_cmh2o=0.0,
        )
        cfg_asym = generator.ScenarioConfig(
            duration_s=20.0, fs_hz=100.0, rr_bpm=15.0,
            ie_ratio=0.4,  # short inspiration (1:1.5)
            cardiogenic_amplitude_cmh2o=0.0,
            measurement_noise_std_cmh2o=0.0,
        )
        sig_sym, _ = generator.synthesize_session(cfg_sym, seed=0)
        sig_asym, _ = generator.synthesize_session(cfg_asym, seed=0)

        # Both must integrate to roughly the same TV (per cycle)
        flow_sym = sig_sym["flow_patient_lpm"]
        flow_asym = sig_asym["flow_patient_lpm"]

        # Inspiration fraction should differ
        insp_frac_sym = float(np.mean(flow_sym > 0))
        insp_frac_asym = float(np.mean(flow_asym > 0))
        assert insp_frac_sym > 0.45 and insp_frac_sym < 0.55  # ~50%
        assert insp_frac_asym < insp_frac_sym - 0.05  # noticeably less
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/yhchoi/Desktop/Claude_Contents/soom/soom_Project
pytest tests/test_signal_processing.py::TestGenerator::test_ie_ratio_changes_inspiration_duration -v
```
Expected: FAIL with `TypeError: ScenarioConfig.__init__() got an unexpected keyword argument 'ie_ratio'`

- [ ] **Step 3: Add ie_ratio field to ScenarioConfig**

`src/signal_processing/generator.py`에서 `ScenarioConfig` 클래스 안, `intentional_leak_lpm` 다음 줄에 추가:
```python
    intentional_leak_lpm: float = 24.0
    blower_rpm_baseline: float = 18000.0
    ie_ratio: float = 1.0  # inspiration_time / total_cycle_time × 2; 1.0 = symmetric sine
```

- [ ] **Step 4: Replace `_breath_flow` with ie_ratio-aware version**

`src/signal_processing/generator.py`에서 `_breath_flow` 함수를 다음으로 교체:
```python
def _breath_flow(t: np.ndarray, rr_bpm: float, tv_ml: float,
                 ie_ratio: float = 1.0) -> np.ndarray:
    """Asymmetric flow waveform with configurable I:E ratio.

    ie_ratio = 1.0 → pure sine (1:1).
    ie_ratio < 1.0 → inspiration shorter than expiration (e.g. 0.5 ≈ 1:2).
    Inspiratory area (∫>0) is preserved equal to TV regardless of ie_ratio.
    """
    f_breath = rr_bpm / 60.0
    period = 1.0 / f_breath
    insp_dur = period * (ie_ratio / (1.0 + ie_ratio)) * 2.0
    insp_dur = float(np.clip(insp_dur, 0.1 * period, 0.9 * period))
    exp_dur = period - insp_dur

    phase = np.mod(t, period)
    flow = np.zeros_like(t)
    insp_mask = phase < insp_dur
    exp_mask = ~insp_mask
    # Half-sine over inspiratory window (positive), half-sine over expiratory (negative)
    flow[insp_mask] = np.sin(np.pi * phase[insp_mask] / insp_dur)
    flow[exp_mask] = -np.sin(np.pi * (phase[exp_mask] - insp_dur) / exp_dur)

    # Scale so ∫(flow>0) dt over one inspiration equals TV (mL → L)
    insp_integral_unit = (2.0 / np.pi) * insp_dur  # ∫sin = 2/π × duration
    peak_flow_lpm = (tv_ml / 1000.0) / insp_integral_unit * 60.0
    return peak_flow_lpm * flow
```

- [ ] **Step 5: Update `synthesize_session` to pass ie_ratio**

`src/signal_processing/generator.py`의 `synthesize_session` 본문에서 `_breath_flow(t, cfg.rr_bpm, cfg.tv_ml)` 호출을 다음으로 교체:
```python
    flow_patient = _breath_flow(t, cfg.rr_bpm, cfg.tv_ml, cfg.ie_ratio)
```

- [ ] **Step 6: Run new test + all existing tests**

```bash
pytest tests/test_signal_processing.py -v
```
Expected: 18 passed (17 original + 1 new). Existing tests must still pass because `ie_ratio=1.0` default = current sine behavior.

- [ ] **Step 7: Commit**

```bash
git add src/signal_processing/generator.py tests/test_signal_processing.py
git commit -m "feat(generator): add ie_ratio for asymmetric inspiration/expiration"
```

---

## Task 2: Generator — `mixed_apneas` (CA → OA 전환)

**Files:**
- Modify: `src/signal_processing/generator.py`
- Test: `tests/test_signal_processing.py`

- [ ] **Step 1: Write failing test**

`TestGenerator` 클래스에 추가:
```python
    def test_mixed_apnea_central_then_obstructive(self):
        """MA = central first half (cardiogenic preserved) → obstructive second half (cardiogenic blocked)."""
        cfg = generator.ScenarioConfig(
            duration_s=80.0, fs_hz=100.0, rr_bpm=15.0,
            mixed_apneas=[(30.0, 20.0, 0.5)],  # at 30s, 20s long, 50% central
            cardiogenic_amplitude_cmh2o=0.4,
            measurement_noise_std_cmh2o=0.02,
        )
        signals, gt = generator.synthesize_session(cfg, seed=3)
        fs = cfg.fs_hz

        # MA event registered in GT
        ma_events = gt.of_type("MA")
        assert len(ma_events) == 1
        assert abs(ma_events[0].start_s - 30.0) < 0.1
        assert abs(ma_events[0].end_s - 50.0) < 0.1

        # Cardiogenic should be present in first half (central) and absent in second (obstructive)
        from signal_processing import filters
        refined = filters.stage2_refine(signals["pressure_cmh2o"], fs_in=fs, fs_out=fs)
        cardio = filters.bandpass_cardiogenic(refined, fs=fs)
        first_half = cardio[int(32 * fs):int(38 * fs)]    # central portion
        second_half = cardio[int(42 * fs):int(48 * fs)]   # obstructive portion
        rms_first = float(np.sqrt(np.mean(first_half ** 2)))
        rms_second = float(np.sqrt(np.mean(second_half ** 2)))
        assert rms_first > rms_second * 1.5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_signal_processing.py::TestGenerator::test_mixed_apnea_central_then_obstructive -v
```
Expected: FAIL with `unexpected keyword argument 'mixed_apneas'`.

- [ ] **Step 3: Add mixed_apneas field**

`ScenarioConfig`의 `hypopneas` 다음 줄에 추가:
```python
    snore_episodes: list[tuple[float, float]] = field(default_factory=list)
    mixed_apneas: list[tuple[float, float, float]] = field(default_factory=list)
    # ^ (start_s, total_duration_s, central_fraction 0~1)
```

- [ ] **Step 4: Process mixed_apneas in synthesize_session**

`synthesize_session` 본문, hypopnea 처리(`# --- 4. Inject hypopneas ---` 블록) 다음에 새 블록 추가:
```python
    # --- 4b. Inject mixed apneas (central → obstructive transition) ---
    for start_s, total_dur, central_fraction in cfg.mixed_apneas:
        central_fraction = float(np.clip(central_fraction, 0.0, 1.0))
        mid_s = start_s + total_dur * central_fraction
        end_s = start_s + total_dur
        # Suppress flow over entire MA window
        flow_patient = _suppress(flow_patient, cfg.fs_hz, start_s, end_s, factor=0.05)
        pressure_mod = pressure - cfg.base_pressure_cmh2o
        pressure_mod = _suppress(pressure_mod, cfg.fs_hz, start_s, end_s, factor=0.05)
        pressure = cfg.base_pressure_cmh2o + pressure_mod
        # Cardiogenic: present during central half, blocked during obstructive half
        cardiogenic_mask = _suppress(
            cardiogenic_mask, cfg.fs_hz, mid_s, end_s, factor=0.05
        )
        gt.events.append(GroundTruthEvent(
            "MA", start_s, end_s,
            metadata={"central_fraction": central_fraction, "transition_s": mid_s},
        ))
```

- [ ] **Step 5: Update GroundTruthEvent EventType literal**

`generator.py` 상단:
```python
EventType = Literal["normal", "OA", "CA", "MA", "hypopnea", "snore", "leak", "csr", "cough"]
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_signal_processing.py -v
```
Expected: 19 passed.

- [ ] **Step 7: Commit**

```bash
git add src/signal_processing/generator.py tests/test_signal_processing.py
git commit -m "feat(generator): add mixed_apneas (central→obstructive transition)"
```

---

## Task 3: Generator — `unintentional_leak` (상수/램프/버스트 프로파일)

**Files:**
- Modify: `src/signal_processing/generator.py`
- Test: `tests/test_signal_processing.py`

- [ ] **Step 1: Write failing test**

`TestGenerator` 클래스에 추가:
```python
    def test_unintentional_leak_ramp_increases_total_flow(self):
        """unintentional_leak_profile='ramp' → total_flow grows linearly over session."""
        cfg = generator.ScenarioConfig(
            duration_s=120.0, fs_hz=100.0, rr_bpm=15.0,
            unintentional_leak_lpm=20.0,
            unintentional_leak_profile="ramp",
            cardiogenic_amplitude_cmh2o=0.0,
            measurement_noise_std_cmh2o=0.0,
        )
        signals, _ = generator.synthesize_session(cfg, seed=0)
        flow_total = signals["flow_lpm"]
        fs = cfg.fs_hz

        # Mean flow over first vs last 20 s should differ by ~20 L/min
        first = float(np.mean(flow_total[int(5 * fs):int(20 * fs)]))
        last = float(np.mean(flow_total[int(100 * fs):int(115 * fs)]))
        assert (last - first) > 10.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_signal_processing.py::TestGenerator::test_unintentional_leak_ramp_increases_total_flow -v
```
Expected: FAIL.

- [ ] **Step 3: Add fields**

`ScenarioConfig`에 추가:
```python
    mixed_apneas: list[tuple[float, float, float]] = field(default_factory=list)
    unintentional_leak_lpm: float = 0.0
    unintentional_leak_profile: str = "constant"  # "constant" | "ramp" | "burst"
```

- [ ] **Step 4: Apply unintentional leak in synthesize_session**

`synthesize_session`의 step 8 (`# --- 8. Total flow including intentional leak ---`)을 다음으로 교체:
```python
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
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_signal_processing.py -v
```
Expected: 20 passed.

- [ ] **Step 6: Commit**

```bash
git add src/signal_processing/generator.py tests/test_signal_processing.py
git commit -m "feat(generator): add unintentional_leak with constant/ramp/burst profiles"
```

---

## Task 4: Generator — `cough_events` (압력 펄스)

**Files:**
- Modify: `src/signal_processing/generator.py`
- Test: `tests/test_signal_processing.py`

- [ ] **Step 1: Write failing test**

```python
    def test_cough_event_creates_pressure_spike(self):
        """Cough adds a brief (~0.5s) sharp pressure spike."""
        cfg = generator.ScenarioConfig(
            duration_s=30.0, fs_hz=100.0, rr_bpm=15.0,
            cough_events=[(15.0, 4.0)],  # 15s, peak 4 cmH2O above baseline
            cardiogenic_amplitude_cmh2o=0.0,
            measurement_noise_std_cmh2o=0.0,
        )
        signals, gt = generator.synthesize_session(cfg, seed=0)
        pressure = signals["pressure_cmh2o"]
        fs = cfg.fs_hz

        baseline_seg = pressure[int(5 * fs):int(10 * fs)]
        cough_seg = pressure[int(14.7 * fs):int(15.4 * fs)]
        assert (np.max(cough_seg) - np.max(baseline_seg)) > 2.0
        assert any(e.type == "cough" for e in gt.events)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_signal_processing.py::TestGenerator::test_cough_event_creates_pressure_spike -v
```
Expected: FAIL.

- [ ] **Step 3: Add field**

`ScenarioConfig`에 추가:
```python
    unintentional_leak_profile: str = "constant"
    cough_events: list[tuple[float, float]] = field(default_factory=list)
    # ^ (start_s, peak_amplitude_cmh2o)
```

- [ ] **Step 4: Inject coughs in synthesize_session**

step 7 (measurement noise) 직전에 추가:
```python
    # --- 6b. Inject cough events (~0.5s gaussian-shaped pressure spike) ---
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
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_signal_processing.py -v
```
Expected: 21 passed.

- [ ] **Step 6: Commit**

```bash
git add src/signal_processing/generator.py tests/test_signal_processing.py
git commit -m "feat(generator): add cough_events with gaussian pressure spikes"
```

---

## Task 5: Generator — 50Hz 전원 간섭

**Files:**
- Modify: `src/signal_processing/generator.py`
- Test: `tests/test_signal_processing.py`

- [ ] **Step 1: Write failing test**

```python
    def test_50hz_interference_appears_in_spectrum(self):
        """power_line_50hz_amplitude > 0 → strong 50 Hz peak in pressure FFT."""
        cfg = generator.ScenarioConfig(
            duration_s=10.0, fs_hz=200.0, rr_bpm=15.0,  # need fs > 100 to see 50Hz
            power_line_50hz_amplitude_cmh2o=0.3,
            cardiogenic_amplitude_cmh2o=0.0,
            measurement_noise_std_cmh2o=0.0,
        )
        signals, _ = generator.synthesize_session(cfg, seed=0)
        from numpy.fft import rfft, rfftfreq
        spec = np.abs(rfft(signals["pressure_cmh2o"]))
        freqs = rfftfreq(len(signals["pressure_cmh2o"]), 1.0 / cfg.fs_hz)
        i_50 = int(np.argmin(np.abs(freqs - 50.0)))
        i_25 = int(np.argmin(np.abs(freqs - 25.0)))
        assert spec[i_50] > spec[i_25] * 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_signal_processing.py::TestGenerator::test_50hz_interference_appears_in_spectrum -v
```
Expected: FAIL.

- [ ] **Step 3: Add field**

`ScenarioConfig`에 추가:
```python
    cough_events: list[tuple[float, float]] = field(default_factory=list)
    power_line_50hz_amplitude_cmh2o: float = 0.0
```

- [ ] **Step 4: Inject 50Hz in synthesize_session**

step 7 (measurement noise) 직전, cough 다음에 추가:
```python
    # --- 6c. Inject 50 Hz power-line interference ---
    if cfg.power_line_50hz_amplitude_cmh2o > 0.0 and 50.0 < cfg.fs_hz / 2.0:
        pressure = pressure + cfg.power_line_50hz_amplitude_cmh2o * np.sin(
            2 * np.pi * 50.0 * t
        )
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_signal_processing.py -v
```
Expected: 22 passed.

- [ ] **Step 6: Commit**

```bash
git add src/signal_processing/generator.py tests/test_signal_processing.py
git commit -m "feat(generator): add 50Hz power-line interference injection"
```

---

## Task 6: Generator — EPR (Expiratory Pressure Relief)

**Files:**
- Modify: `src/signal_processing/generator.py`
- Test: `tests/test_signal_processing.py`

- [ ] **Step 1: Write failing test**

```python
    def test_epr_drops_pressure_during_expiration(self):
        """EPR active → expiratory mean pressure < inspiratory mean by ≈ epr_relief."""
        cfg = generator.ScenarioConfig(
            duration_s=20.0, fs_hz=100.0, rr_bpm=15.0,
            base_pressure_cmh2o=10.0,
            epr_enabled=True,
            epr_relief_cmh2o=2.0,
            cardiogenic_amplitude_cmh2o=0.0,
            measurement_noise_std_cmh2o=0.0,
        )
        signals, _ = generator.synthesize_session(cfg, seed=0)
        flow = signals["flow_patient_lpm"]
        pressure = signals["pressure_cmh2o"]
        insp_mask = flow > 0
        exp_mask = flow < 0
        mean_insp = float(np.mean(pressure[insp_mask]))
        mean_exp = float(np.mean(pressure[exp_mask]))
        assert (mean_insp - mean_exp) > 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_signal_processing.py::TestGenerator::test_epr_drops_pressure_during_expiration -v
```
Expected: FAIL.

- [ ] **Step 3: Add fields**

`ScenarioConfig`에 추가:
```python
    power_line_50hz_amplitude_cmh2o: float = 0.0
    epr_enabled: bool = False
    epr_relief_cmh2o: float = 0.0
```

- [ ] **Step 4: Apply EPR in synthesize_session**

step 1 (`flow_patient = _breath_flow(...)` 다음, pressure 계산 직후) 위치에 EPR 적용:

`synthesize_session` 안 step 1 블록을 다음으로 교체:
```python
    # --- 1. Base normal breathing ---
    flow_patient = _breath_flow(t, cfg.rr_bpm, cfg.tv_ml, cfg.ie_ratio)
    pressure = _flow_to_pressure(flow_patient, cfg.base_pressure_cmh2o)
    # EPR: drop pressure during expiration (flow < 0)
    if cfg.epr_enabled and cfg.epr_relief_cmh2o > 0.0:
        epr_drop = np.where(flow_patient < 0, cfg.epr_relief_cmh2o, 0.0)
        # Smooth transition: 0.1 s rolling mean
        win = max(int(0.1 * cfg.fs_hz), 1)
        kernel = np.ones(win) / win
        epr_drop = np.convolve(epr_drop, kernel, mode="same")
        pressure = pressure - epr_drop
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_signal_processing.py -v
```
Expected: 23 passed.

- [ ] **Step 6: Commit**

```bash
git add src/signal_processing/generator.py tests/test_signal_processing.py
git commit -m "feat(generator): add EPR (expiratory pressure relief) modeling"
```

---

## Task 7: simulator/__init__.py + ui_state.py (위젯 dict → ScenarioConfig)

**Files:**
- Create: `src/simulator/__init__.py`
- Create: `src/simulator/ui_state.py`
- Create: `tests/test_simulator.py`

- [ ] **Step 1: Write failing test**

새 파일 `tests/test_simulator.py`:
```python
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
        # Collect all event intervals
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
        """Asking for events whose total duration exceeds session → silently clip."""
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
        # Total OA budget would be 20×30 = 600s but session is 60s → must clip
        assert len(cfg.obstructive_apneas) < 20
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_simulator.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'simulator'`.

- [ ] **Step 3: Create simulator package**

새 파일 `src/simulator/__init__.py`:
```python
"""Streamlit simulator UI helpers — converts widget state to ScenarioConfig,
randomizes parameters, packages exports, and builds plotly figures."""
from __future__ import annotations

from . import ui_state, randomizer, exporter, plotting

__all__ = ["ui_state", "randomizer", "exporter", "plotting"]
```

- [ ] **Step 4: Create ui_state.py**

새 파일 `src/simulator/ui_state.py`:
```python
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
    """Place n_events of approximate mean duration without overlapping `occupied`."""
    placed: list[tuple[float, float]] = []
    if duration_s < margin_s * 2:
        return placed
    for _ in range(n_events):
        # Jitter duration ±20%
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
        intervals = _allocate_event_intervals(
            n, mean_dur, duration_s, occupied, rng,
        )
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
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_simulator.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/simulator/ tests/test_simulator.py
git commit -m "feat(simulator): widget-dict to ScenarioConfig converter with non-overlap event allocation"
```

---

## Task 8: simulator/randomizer.py (🎲 임상 정상 범위 샘플링)

**Files:**
- Create: `src/simulator/randomizer.py`
- Test: `tests/test_simulator.py` (TestRandomizer 추가)

- [ ] **Step 1: Write failing test**

`tests/test_simulator.py` 끝에 추가:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_simulator.py::TestRandomizer -v
```
Expected: FAIL with import error.

- [ ] **Step 3: Create randomizer.py**

새 파일 `src/simulator/randomizer.py`:
```python
"""Randomize slider values within clinically plausible ranges."""
from __future__ import annotations

import numpy as np


def randomize_widgets(seed: int) -> dict:
    """Sample a full set of slider values from clinical normal ranges.

    Returns a dict in the same shape as the UI widget state, ready to feed
    into ui_state.build_scenario_config().
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
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_simulator.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/simulator/randomizer.py tests/test_simulator.py
git commit -m "feat(simulator): randomizer for clinical-range slider sampling"
```

---

## Task 9: simulator/exporter.py (CSV+JSON+ZIP)

**Files:**
- Create: `src/simulator/exporter.py`
- Test: `tests/test_simulator.py` (TestExporter 추가)

- [ ] **Step 1: Write failing test**

`tests/test_simulator.py`에 추가:
```python
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
        n_data = len(lines) - 1  # minus header
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_simulator.py::TestExporter -v
```
Expected: FAIL.

- [ ] **Step 3: Create exporter.py**

새 파일 `src/simulator/exporter.py`:
```python
"""Package synthesized signals + ground truth + metadata into a ZIP archive."""
from __future__ import annotations

import io
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from signal_processing.generator import GroundTruth, ScenarioConfig

SCHEMA_VERSION = "1.0"

_README = """soom_Project Synthetic CPAP Pressure Signal — Validation Dataset
================================================================
1. Read signal.csv into your analysis tool
   → Use 'pressure_cmh2o' column as the input signal (fs given in metadata.json)
2. Run your apnea/hypopnea detector on that single column
3. Compare your detections with ground_truth.csv
   (event_type, start_s, end_s) tuples are the labels
4. metadata.json describes the synthesis parameters

Note: 'flow_lpm' / 'flow_patient_lpm' are GROUND TRUTH only.
A real CPAP device with a single ΔP sensor cannot directly measure flow.
"""


def _git_short_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _signal_dataframe(signals: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "time_s": signals["t_s"],
        "pressure_cmh2o": signals["pressure_cmh2o"],
        "flow_lpm": signals["flow_lpm"],
        "flow_patient_lpm": signals["flow_patient_lpm"],
        "blower_rpm": signals["blower_rpm"],
    })


def _ground_truth_dataframe(gt: GroundTruth) -> pd.DataFrame:
    rows = []
    for ev in gt.events:
        meta = dict(ev.metadata) if ev.metadata else {}
        rows.append({
            "event_type": ev.type,
            "start_s": ev.start_s,
            "end_s": ev.end_s,
            "duration_s": ev.duration_s,
            "severity": meta.pop("severity", ""),
            "metadata": json.dumps(meta) if meta else "",
        })
    return pd.DataFrame(rows, columns=[
        "event_type", "start_s", "end_s", "duration_s", "severity", "metadata",
    ])


def _build_metadata(cfg: ScenarioConfig, gt: GroundTruth, seed: int) -> dict:
    counts: dict[str, int] = {}
    for ev in gt.events:
        counts[ev.type] = counts.get(ev.type, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "seed": int(seed),
        "fs_hz": float(cfg.fs_hz),
        "duration_s": float(cfg.duration_s),
        "scenario": {
            "rr_bpm": float(cfg.rr_bpm),
            "tv_ml": float(cfg.tv_ml),
            "ie_ratio": float(cfg.ie_ratio),
            "base_pressure_cmh2o": float(cfg.base_pressure_cmh2o),
            "intentional_leak_lpm": float(cfg.intentional_leak_lpm),
            "hr_bpm": float(cfg.heart_rate_bpm),
            "cardiogenic_amplitude_cmh2o": float(cfg.cardiogenic_amplitude_cmh2o),
            "measurement_noise_std_cmh2o": float(cfg.measurement_noise_std_cmh2o),
            "epr_enabled": bool(cfg.epr_enabled),
            "epr_relief_cmh2o": float(cfg.epr_relief_cmh2o),
            "power_line_50hz_amplitude_cmh2o":
                float(cfg.power_line_50hz_amplitude_cmh2o),
            "unintentional_leak_lpm": float(cfg.unintentional_leak_lpm),
            "unintentional_leak_profile": str(cfg.unintentional_leak_profile),
        },
        "event_summary": counts,
        "soom_version": _git_short_hash(),
        "spec_reference": "03_신호처리_사양서.md v0.4 §0.6",
    }


def build_zip(signals: dict, gt: GroundTruth, cfg: ScenarioConfig,
              seed: int) -> bytes:
    """Return ZIP bytes containing signal.csv + ground_truth.csv + metadata.json + README.txt."""
    sig_df = _signal_dataframe(signals)
    gt_df = _ground_truth_dataframe(gt)
    meta = _build_metadata(cfg, gt, seed)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("signal.csv", sig_df.to_csv(index=False, lineterminator="\n"))
        z.writestr("ground_truth.csv",
                   gt_df.to_csv(index=False, lineterminator="\n"))
        z.writestr("metadata.json", json.dumps(meta, indent=2, ensure_ascii=False))
        z.writestr("README.txt", _README)
    return buf.getvalue()


def filename_for(seed: int, ts: datetime | None = None) -> str:
    ts = ts or datetime.now()
    return f"soom_simulator_{ts.strftime('%Y-%m-%d_%H%M%S')}_{int(seed)}.zip"
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_simulator.py -v
```
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/simulator/exporter.py tests/test_simulator.py
git commit -m "feat(simulator): exporter — CSV+JSON+README ZIP for external validation"
```

---

## Task 10: simulator/plotting.py (plotly 멀티패널)

**Files:**
- Create: `src/simulator/plotting.py`
- Test: `tests/test_simulator.py` (TestPlotting 추가)

- [ ] **Step 1: Write failing test**

`tests/test_simulator.py`에 추가:
```python
from simulator import plotting  # noqa: E402


class TestPlotting:
    def test_build_figure_basic_returns_two_subplots(self):
        cfg = generator.default_demo_scenario()
        signals, gt = generator.synthesize_session(cfg, seed=42)
        fig = plotting.build_figure(signals, gt, show_advanced=False)
        # Expect 2 traces minimum (pressure, flow)
        assert len(fig.data) >= 2

    def test_build_figure_advanced_returns_more_subplots(self):
        cfg = generator.default_demo_scenario()
        signals, gt = generator.synthesize_session(cfg, seed=42)
        fig_basic = plotting.build_figure(signals, gt, show_advanced=False)
        fig_adv = plotting.build_figure(signals, gt, show_advanced=True)
        assert len(fig_adv.data) > len(fig_basic.data)

    def test_event_shapes_count_matches_gt(self):
        cfg = generator.default_demo_scenario()
        signals, gt = generator.synthesize_session(cfg, seed=42)
        fig = plotting.build_figure(signals, gt, show_advanced=False)
        # One vrect shape per event
        assert len(fig.layout.shapes) >= len(gt.events)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_simulator.py::TestPlotting -v
```
Expected: FAIL.

- [ ] **Step 3: Create plotting.py**

새 파일 `src/simulator/plotting.py`:
```python
"""Multi-panel plotly figure for the Signal Simulator page."""
from __future__ import annotations

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from signal_processing.generator import GroundTruth

EVENT_COLOR = {
    "OA": "#E11D2C",
    "CA": "#1F77B4",
    "MA": "#9333EA",
    "hypopnea": "#F3B100",
    "snore": "#16A34A",
    "cough": "#FB923C",
    "csr": "#7C3AED",
}


def _add_event_shapes(fig, gt: GroundTruth) -> None:
    for ev in gt.events:
        color = EVENT_COLOR.get(ev.type, "#888")
        fig.add_vrect(
            x0=ev.start_s, x1=ev.end_s,
            fillcolor=color, opacity=0.18,
            line_width=0, layer="below",
            annotation_text=ev.type, annotation_position="top left",
            annotation_font_size=10,
        )


def build_figure(signals: dict, gt: GroundTruth,
                 show_advanced: bool = False) -> go.Figure:
    """Build the Signal Simulator main plot.

    Basic mode: 2 panels (pressure, flow).
    Advanced mode: 4 panels (pressure, flow, RPM, 4-branch overlay).
    """
    n_panels = 4 if show_advanced else 2
    titles = ["Pressure (cmH₂O)", "Flow ground truth (L/min)"]
    if show_advanced:
        titles += ["Blower RPM", "4-branch BPF (overlay)"]

    fig = make_subplots(
        rows=n_panels, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=titles,
    )
    t = signals["t_s"]

    fig.add_trace(go.Scatter(x=t, y=signals["pressure_cmh2o"],
                             name="pressure", line=dict(color="#1B6FB8", width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=signals["flow_patient_lpm"],
                             name="flow_patient", line=dict(color="#16A34A", width=1)),
                  row=2, col=1)

    if show_advanced:
        fig.add_trace(go.Scatter(x=t, y=signals["blower_rpm"],
                                 name="rpm", line=dict(color="#9333EA", width=1)),
                      row=3, col=1)
        # 4-branch BPF — compute on the fly
        from signal_processing import filters
        fs = float(1.0 / np.mean(np.diff(t)))
        refined = filters.stage2_refine(signals["pressure_cmh2o"],
                                        fs_in=fs, fs_out=fs)
        branches = filters.apply_4branch_bpf(refined, fs=fs)
        for name, color in [("breath", "#1B6FB8"), ("cardiogenic", "#E11D2C"),
                            ("fot", "#F3B100"), ("snore", "#16A34A")]:
            fig.add_trace(go.Scatter(x=t, y=branches[name],
                                     name=name, line=dict(color=color, width=1)),
                          row=4, col=1)

    _add_event_shapes(fig, gt)
    fig.update_layout(
        height=200 * n_panels + 80,
        showlegend=True,
        margin=dict(l=60, r=20, t=40, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time (s)", row=n_panels, col=1)
    return fig
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_simulator.py -v
```
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/simulator/plotting.py tests/test_simulator.py
git commit -m "feat(simulator): plotly multi-panel figure with event shading"
```

---

## Task 11: pages/2_📈_Signal_Simulator.py (Streamlit 페이지)

**Files:**
- Create: `pages/2_📈_Signal_Simulator.py`

- [ ] **Step 1: Verify Streamlit multipage convention by checking existing page**

```bash
ls pages/ 2>/dev/null || ls *.py | head -20
```
Expected: existing Streamlit setup. If `pages/` 폴더가 없다면 다음 step에서 함께 생성.

- [ ] **Step 2: Create the page file**

새 파일 `pages/2_📈_Signal_Simulator.py`:
```python
"""Signal Simulator — interactive page to synthesize CPAP pressure raw signals.

See docs/specs/2026-04-29-signal-simulator-design.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from signal_processing import generator  # noqa: E402
from simulator import exporter, plotting, randomizer, ui_state  # noqa: E402

st.set_page_config(page_title="Signal Simulator", page_icon="📈", layout="wide")
st.title("📈 Signal Simulator — 양압기 압력 신호 합성")

# --------------------------------------------------------------------------
# Session state init
# --------------------------------------------------------------------------
if "sim_widgets" not in st.session_state:
    st.session_state["sim_widgets"] = {
        "duration_s": 300.0, "fs_hz": 100.0,
        "rr_bpm": 15.0, "tv_ml": 500.0, "ie_ratio": 0.5,
        "base_pressure_cmh2o": 9.5,
        "epr_enabled": False, "epr_relief_cmh2o": 1.5,
        "n_oa": 2, "n_ca": 1, "n_ma": 0, "n_hypopnea": 1,
        "n_snore": 1, "n_cough": 0,
        "oa_mean_dur_s": 18.0, "ca_mean_dur_s": 15.0,
        "hypopnea_mean_dur_s": 20.0,
        "hr_bpm": 70.0, "cardiogenic_amplitude_cmh2o": 0.25,
        "measurement_noise_std_cmh2o": 0.05,
        "power_line_50hz_enabled": False,
        "unintentional_leak_lpm": 0.0,
    }
if "pinned_seed" not in st.session_state:
    st.session_state["pinned_seed"] = 42

# --------------------------------------------------------------------------
# Sidebar widgets
# --------------------------------------------------------------------------
W = st.session_state["sim_widgets"]

with st.sidebar:
    st.header("⚙️ 시나리오 설정")

    if st.button("🎲 Randomize", use_container_width=True):
        new_widgets = randomizer.randomize_widgets(seed=int(time.time()))
        st.session_state["sim_widgets"].update(new_widgets)
        st.rerun()
    if st.button("↻ Reset to defaults", use_container_width=True):
        for k, v in {
            "duration_s": 300.0, "rr_bpm": 15.0, "tv_ml": 500.0, "ie_ratio": 0.5,
            "base_pressure_cmh2o": 9.5, "epr_enabled": False, "epr_relief_cmh2o": 1.5,
            "n_oa": 2, "n_ca": 1, "n_ma": 0, "n_hypopnea": 1,
            "n_snore": 1, "n_cough": 0,
            "oa_mean_dur_s": 18.0, "ca_mean_dur_s": 15.0, "hypopnea_mean_dur_s": 20.0,
            "hr_bpm": 70.0, "cardiogenic_amplitude_cmh2o": 0.25,
            "measurement_noise_std_cmh2o": 0.05,
            "power_line_50hz_enabled": False, "unintentional_leak_lpm": 0.0,
        }.items():
            W[k] = v
        st.rerun()

    pin_seed = st.checkbox("🎯 Pin seed", value=False)

    with st.expander("세션", expanded=True):
        W["duration_s"] = st.slider("Duration (s)", 60, 1800,
                                    int(W["duration_s"]), step=30)
        st.caption(f"fs = {W['fs_hz']:.0f} Hz (read-only)")

    with st.expander("호흡 패턴", expanded=True):
        W["rr_bpm"] = st.slider("RR (bpm)", 8.0, 30.0, float(W["rr_bpm"]), 0.5)
        W["tv_ml"] = st.slider("TV (mL)", 200, 800, int(W["tv_ml"]), 10)
        W["ie_ratio"] = st.slider("I:E ratio", 0.3, 1.0, float(W["ie_ratio"]), 0.05)

    with st.expander("압력 시스템"):
        W["base_pressure_cmh2o"] = st.slider("Base pressure (cmH₂O)", 4.0, 20.0,
                                              float(W["base_pressure_cmh2o"]), 0.5)
        W["epr_enabled"] = st.checkbox("EPR (호기 압력 완화) 활성", value=W["epr_enabled"])
        if W["epr_enabled"]:
            W["epr_relief_cmh2o"] = st.slider("EPR relief (cmH₂O)", 0.5, 3.0,
                                              float(W["epr_relief_cmh2o"]), 0.1)

    with st.expander("이벤트 — 개수", expanded=True):
        W["n_oa"] = st.slider("OA (Obstructive Apnea)", 0, 20, int(W["n_oa"]))
        W["n_ca"] = st.slider("CA (Central Apnea)", 0, 10, int(W["n_ca"]))
        W["n_ma"] = st.slider("MA (Mixed Apnea)", 0, 5, int(W["n_ma"]))
        W["n_hypopnea"] = st.slider("Hypopnea", 0, 20, int(W["n_hypopnea"]))
        W["n_snore"] = st.slider("Snore episodes", 0, 10, int(W["n_snore"]))
        W["n_cough"] = st.slider("Cough", 0, 10, int(W["n_cough"]))

    with st.expander("이벤트 — 평균 지속시간"):
        W["oa_mean_dur_s"] = st.slider("OA dur (s)", 10.0, 60.0,
                                        float(W["oa_mean_dur_s"]), 1.0)
        W["ca_mean_dur_s"] = st.slider("CA dur (s)", 10.0, 60.0,
                                        float(W["ca_mean_dur_s"]), 1.0)
        W["hypopnea_mean_dur_s"] = st.slider("Hypopnea dur (s)", 10.0, 60.0,
                                              float(W["hypopnea_mean_dur_s"]), 1.0)

    with st.expander("심혈관"):
        W["hr_bpm"] = st.slider("Heart rate (bpm)", 40.0, 100.0,
                                float(W["hr_bpm"]), 1.0)
        W["cardiogenic_amplitude_cmh2o"] = st.slider(
            "Cardiogenic amplitude (cmH₂O)", 0.0, 0.5,
            float(W["cardiogenic_amplitude_cmh2o"]), 0.01)

    with st.expander("노이즈/누설"):
        W["measurement_noise_std_cmh2o"] = st.slider(
            "Measurement noise σ (cmH₂O)", 0.0, 0.2,
            float(W["measurement_noise_std_cmh2o"]), 0.01)
        W["power_line_50hz_enabled"] = st.checkbox(
            "50 Hz power-line interference", value=W["power_line_50hz_enabled"])
        W["unintentional_leak_lpm"] = st.slider(
            "Unintentional leak (L/min)", 0.0, 30.0,
            float(W["unintentional_leak_lpm"]), 1.0)

# --------------------------------------------------------------------------
# Validation warnings
# --------------------------------------------------------------------------
mv_lpm = (W["rr_bpm"] * W["tv_ml"]) / 1000.0
if mv_lpm > 25.0:
    st.warning(f"비현실적 분당환기량: {mv_lpm:.1f} L/min (>25). 진행하지만 결과는 부자연스러울 수 있습니다.")
if W["duration_s"] > 600:
    st.info(f"⏱ 큰 신호 — 약 {W['duration_s'] * W['fs_hz'] * 1e-6 * 50:.1f}초 생성 시간 예상.")

# --------------------------------------------------------------------------
# Generate signals (cached)
# --------------------------------------------------------------------------
seed = (int(st.session_state["pinned_seed"]) if pin_seed
        else int(time.time() * 1000) % (2**32))


@st.cache_data(max_entries=20, show_spinner=False)
def generate_signals_cached(widget_items: tuple, seed: int):
    widgets = dict(widget_items)
    cfg = ui_state.build_scenario_config(widgets, seed=seed)
    signals, gt = generator.synthesize_session(cfg, seed=seed)
    return cfg, signals, gt


widget_items_for_cache = tuple(sorted(W.items()))
try:
    cfg, signals, gt = generate_signals_cached(widget_items_for_cache, seed)
except Exception as e:
    st.error(f"신호 생성 실패: {e}")
    st.stop()

# --------------------------------------------------------------------------
# Main panel — plot + meta + export
# --------------------------------------------------------------------------
show_advanced = st.toggle("▼ 고급 보기 (RPM, 4-branch BPF)", value=False)

fig = plotting.build_figure(signals, gt, show_advanced=show_advanced)
st.plotly_chart(fig, use_container_width=True)

# Auto-random meta panel
with st.expander("📋 자동 랜덤 메타 (read-only)", expanded=False):
    st.write(f"**Seed**: `{seed}`")
    oa_starts = [round(s, 1) for s, _ in cfg.obstructive_apneas]
    ca_starts = [round(s, 1) for s, _ in cfg.central_apneas]
    hyps_sev = [round(sev, 2) for _, _, sev in cfg.hypopneas]
    st.write(f"**OA starts (s)**: {oa_starts}")
    st.write(f"**CA starts (s)**: {ca_starts}")
    st.write(f"**Hypopnea severities**: {hyps_sev}")
    if cfg.mixed_apneas:
        st.write(f"**MA central fraction**: "
                 f"{[round(f, 2) for _, _, f in cfg.mixed_apneas]}")

# Export
st.subheader("📥 Export")
zip_bytes = exporter.build_zip(signals, gt, cfg, seed=seed)
fname = exporter.filename_for(seed=seed)
st.download_button(
    label=f"⬇ Download ZIP ({len(zip_bytes) / 1024:.0f} KB)",
    data=zip_bytes,
    file_name=fname,
    mime="application/zip",
    use_container_width=True,
)
st.caption("ZIP 내부: signal.csv, ground_truth.csv, metadata.json, README.txt")
```

- [ ] **Step 3: Run Streamlit and manually smoke-test**

```bash
streamlit run streamlit_app.py
```

Then in browser, switch to "Signal Simulator" page and verify:
1. Page loads, sidebar shows all expanders
2. Default scenario plot shows pressure + flow with event shading (2 OA, 1 CA, 1 hypopnea, 1 snore)
3. Move "Duration" slider → plot redraws (<500 ms for 5 min/100 Hz)
4. Toggle "고급 보기" → 4 panels appear (RPM, 4-branch)
5. Click 🎲 Randomize → all sliders + plot + meta update
6. Click "Download ZIP" → file downloads
7. Unzip → `pandas.read_csv("signal.csv")` works

- [ ] **Step 4: Commit**

```bash
git add pages/2_📈_Signal_Simulator.py
git commit -m "feat(pages): Signal Simulator Streamlit page (sliders + Randomize + ZIP export)"
```

---

## Task 12: README + 문서 동기화

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add Signal Simulator section to README**

`README.md`의 "신호처리 PoC 데모" 섹션 끝(`pytest tests/test_signal_processing.py -v` 코드블록 다음)에 추가:
```markdown

### Signal Simulator (인터랙티브 합성 도구)

`docs/specs/2026-04-29-signal-simulator-design.md` 참조.

```bash
streamlit run streamlit_app.py
# → 브라우저에서 "📈 Signal Simulator" 페이지 선택
```

기능:
- 슬라이더로 duration, RR, TV, OA·CA·MA·hypopnea·snore·cough 횟수, 심박수, 노이즈 등 직접 조작
- 🎲 Randomize 버튼으로 임상 정상 범위 내 자동 시나리오 생성
- 외부 알고리즘 검증용 ZIP 다운로드 (signal.csv + ground_truth.csv + metadata.json + README.txt)

테스트:
```bash
pytest tests/test_simulator.py -v
```
```

- [ ] **Step 2: Verify all tests pass**

```bash
pytest tests/ -v
```
Expected: 6(generator new) + 17(existing signal) + 13(simulator) = 36 passed.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add Signal Simulator section to README"
```

---

## Self-Review

**1. Spec coverage check**:
- §2.3 generator Tier 2 확장 (ie_ratio, MA, unintentional_leak, cough, 50Hz, EPR) → Tasks 1~6 ✓
- §2.1/2.2 simulator 모듈 4개 → Tasks 7~10 ✓
- §3 UI 레이아웃, §4.1~4.3 데이터 흐름·캐시·시드 → Task 11 ✓
- §4.4 Export ZIP 명세 → Task 9 ✓
- §5.1 오류 처리 (warning, st.error) → Task 11 ✓
- §5.2 테스트 (TestUIState/TestRandomizer/TestExporter/TestPlotting/Tier2) → Tasks 1~10 분산 ✓
- §5.3 수동 smoke test → Task 11 Step 3 ✓

**2. Placeholder scan**: 모든 step에 실제 코드 또는 명확한 명령. "TBD"/"적절한"/"비슷한" 표현 없음 ✓

**3. Type consistency**:
- `widgets` dict 키들: `n_oa`/`n_ca`/`n_ma`/`n_hypopnea`/`n_snore`/`n_cough` — Task 7, 8, 11 모두 동일 ✓
- `ScenarioConfig` 필드 추가 순서: ie_ratio (T1) → mixed_apneas (T2) → unintentional_leak_lpm/_profile (T3) → cough_events (T4) → power_line_50hz_amplitude_cmh2o (T5) → epr_enabled/epr_relief_cmh2o (T6) ✓
- `build_zip(signals, gt, cfg, seed)` — Task 9 정의, Task 11 호출 시그니처 동일 ✓
- `build_figure(signals, gt, show_advanced)` — Task 10 정의, Task 11 호출 동일 ✓

이상 self-review 완료, 이슈 없음.

---

## 실행 옵션

Plan complete and saved to `docs/plans/2026-04-29-signal-simulator.md`. 두 가지 실행 옵션:

**1. Subagent-Driven (recommended)** — 매 task마다 fresh subagent 디스패치, task 사이 리뷰, 빠른 반복

**2. Inline Execution** — 이 세션에서 executing-plans 스킬로 batch 실행, checkpoint 사이 리뷰

어느 쪽으로 진행하시겠습니까?
