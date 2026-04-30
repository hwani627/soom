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
        "intentional_leak_lpm": 24.0,
    }
if "pinned_seed" not in st.session_state:
    st.session_state["pinned_seed"] = 42

# --------------------------------------------------------------------------
# Sidebar widgets
# --------------------------------------------------------------------------
W = st.session_state["sim_widgets"]

with st.sidebar:
    st.header("⚙️ 시나리오 설정")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎲 Randomize", width="stretch"):
            new_widgets = randomizer.randomize_widgets(seed=int(time.time()))
            st.session_state["sim_widgets"].update(new_widgets)
            st.rerun()
    with col2:
        if st.button("↻ Reset", width="stretch"):
            for k, v in {
                "duration_s": 300.0, "rr_bpm": 15.0, "tv_ml": 500.0, "ie_ratio": 0.5,
                "base_pressure_cmh2o": 9.5, "epr_enabled": False, "epr_relief_cmh2o": 1.5,
                "n_oa": 2, "n_ca": 1, "n_ma": 0, "n_hypopnea": 1,
                "n_snore": 1, "n_cough": 0,
                "oa_mean_dur_s": 18.0, "ca_mean_dur_s": 15.0, "hypopnea_mean_dur_s": 20.0,
                "hr_bpm": 70.0, "cardiogenic_amplitude_cmh2o": 0.25,
                "measurement_noise_std_cmh2o": 0.05,
                "power_line_50hz_enabled": False, "unintentional_leak_lpm": 0.0,
                "intentional_leak_lpm": 24.0,
            }.items():
                W[k] = v
            st.rerun()

    pin_seed = st.checkbox("🎯 Pin seed (재현 고정)", value=False)

    with st.expander("세션", expanded=True):
        W["duration_s"] = st.slider("Duration (s)", 60, 1800, int(W["duration_s"]), step=30)
        st.caption(f"fs = {W['fs_hz']:.0f} Hz  |  samples = {int(W['duration_s'] * W['fs_hz']):,}")

    with st.expander("호흡 패턴", expanded=True):
        W["rr_bpm"] = st.slider("RR (bpm)", 8.0, 30.0, float(W["rr_bpm"]), 0.5)
        W["tv_ml"] = st.slider("TV (mL)", 200, 800, int(W["tv_ml"]), 10)
        W["ie_ratio"] = st.slider("I:E ratio", 0.3, 1.0, float(W["ie_ratio"]), 0.05,
                                   help="흡기/호기 비율. 0.5 = 1:2 (정상 성인)")

    with st.expander("압력 시스템"):
        W["base_pressure_cmh2o"] = st.slider("Base pressure (cmH₂O)", 4.0, 20.0,
                                              float(W["base_pressure_cmh2o"]), 0.5)
        W["epr_enabled"] = st.checkbox("EPR 활성 (호기 압력 완화)", value=bool(W["epr_enabled"]))
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
        W["oa_mean_dur_s"] = st.slider("OA dur (s)", 10.0, 60.0, float(W["oa_mean_dur_s"]), 1.0)
        W["ca_mean_dur_s"] = st.slider("CA dur (s)", 10.0, 60.0, float(W["ca_mean_dur_s"]), 1.0)
        W["hypopnea_mean_dur_s"] = st.slider("Hypopnea dur (s)", 10.0, 60.0,
                                              float(W["hypopnea_mean_dur_s"]), 1.0)

    with st.expander("심혈관"):
        W["hr_bpm"] = st.slider("Heart rate (bpm)", 40.0, 100.0, float(W["hr_bpm"]), 1.0)
        W["cardiogenic_amplitude_cmh2o"] = st.slider(
            "Cardiogenic amplitude (cmH₂O)", 0.0, 0.5,
            float(W["cardiogenic_amplitude_cmh2o"]), 0.01)

    with st.expander("노이즈/누설"):
        W["measurement_noise_std_cmh2o"] = st.slider(
            "Measurement noise σ (cmH₂O)", 0.0, 0.2,
            float(W["measurement_noise_std_cmh2o"]), 0.01)
        W["power_line_50hz_enabled"] = st.checkbox(
            "50 Hz power-line interference", value=bool(W["power_line_50hz_enabled"]))
        W["unintentional_leak_lpm"] = st.slider(
            "Unintentional leak (L/min)", 0.0, 30.0, float(W["unintentional_leak_lpm"]), 1.0)
        W["intentional_leak_lpm"] = st.slider(
            "Intentional leak (L/min)", 15.0, 45.0, float(W["intentional_leak_lpm"]), 1.0)

# --------------------------------------------------------------------------
# Validation warnings
# --------------------------------------------------------------------------
mv_lpm = (W["rr_bpm"] * W["tv_ml"]) / 1000.0
if mv_lpm > 25.0:
    st.warning(f"⚠️ 비현실적 분당환기량: {mv_lpm:.1f} L/min (>25). 진행하지만 결과는 부자연스러울 수 있습니다.")
if W["duration_s"] > 600:
    est_s = W["duration_s"] * W["fs_hz"] * 1e-6 * 50
    st.info(f"⏱ 큰 신호 — 약 {est_s:.1f}초 생성 시간 예상.")

# --------------------------------------------------------------------------
# Generate signals (cached)
# --------------------------------------------------------------------------
seed = (int(st.session_state["pinned_seed"]) if pin_seed
        else int(time.time() * 1000) % (2 ** 32))


@st.cache_data(max_entries=20, show_spinner="신호 생성 중...")
def _generate(widget_items: tuple, seed: int):
    widgets = dict(widget_items)
    cfg = ui_state.build_scenario_config(widgets, seed=seed)
    signals, gt = generator.synthesize_session(cfg, seed=seed)
    return cfg, signals, gt


widget_items_for_cache = tuple(sorted(W.items()))
try:
    cfg, signals, gt = _generate(widget_items_for_cache, seed)
except Exception as exc:
    st.error(f"신호 생성 실패: {exc}")
    st.stop()

# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------
show_advanced = st.toggle("▼ 고급 보기 (RPM · 4-branch BPF)", value=False)

fig = plotting.build_figure(signals, gt, show_advanced=show_advanced)
st.plotly_chart(fig, width="stretch")

# Auto-random meta panel
with st.expander("📋 자동 랜덤 메타 (read-only)", expanded=False):
    st.write(f"**Seed:** `{seed}`")
    st.write(f"**OA starts (s):** {[round(s, 1) for s, _ in cfg.obstructive_apneas]}")
    st.write(f"**CA starts (s):** {[round(s, 1) for s, _ in cfg.central_apneas]}")
    if cfg.hypopneas:
        st.write(f"**Hypopnea severities:** {[round(sev, 2) for _, _, sev in cfg.hypopneas]}")
    if cfg.mixed_apneas:
        st.write(f"**MA central fractions:** {[round(f, 2) for _, _, f in cfg.mixed_apneas]}")

# Export
st.subheader("📥 Export")
zip_bytes = exporter.build_zip(signals, gt, cfg, seed=seed)
fname = exporter.filename_for(seed=seed)
st.download_button(
    label=f"⬇ Download ZIP  ({len(zip_bytes) / 1024:.0f} KB)",
    data=zip_bytes,
    file_name=fname,
    mime="application/zip",
    width="stretch",
)
st.caption("ZIP 내부: `signal.csv` · `ground_truth.csv` · `metadata.json` · `README.txt`")
