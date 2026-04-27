"""Soom CPAP Quality Analyzer — Streamlit demo app.

Run locally:    streamlit run app.py
Deploy:         Streamlit Community Cloud (https://share.streamlit.io)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src import detector, evaluator, loader, plotting

REPO_ROOT = Path(__file__).parent
DATA_ROOT = REPO_ROOT / "CAPA_Data"

st.set_page_config(
    page_title="Soom CPAP Quality Analyzer",
    page_icon="🌙",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
st.sidebar.title("🌙 Soom Analyzer")
st.sidebar.caption("PSG-Level CPAP Demo · v0.2.0")

available_dates = loader.list_available_dates(DATA_ROOT)
if not available_dates:
    st.sidebar.error(f"sleephq_*.csv 파일을 찾을 수 없습니다: {DATA_ROOT}")
    st.stop()

date = st.sidebar.selectbox("날짜 선택", options=available_dates, index=0)

st.sidebar.divider()
st.sidebar.markdown("**검출 알고리즘**")
show_aasm   = st.sidebar.checkbox("AASM (≥90% / ≥30%)", value=True)
show_resmed = st.sidebar.checkbox("ResMed (<25% / <50%)", value=True)
show_sleephq = st.sidebar.checkbox("SleepHQ 라벨 (실제 이벤트)", value=True)

with st.sidebar.expander("ResMed 임계값 슬라이더"):
    resmed_apnea_thr = st.slider("Apnea 임계 (ratio)",   0.05, 0.50, 0.25, step=0.01)
    resmed_hypop_thr = st.slider("Hypopnea 임계 (ratio)", 0.30, 0.90, 0.50, step=0.01)
    min_dur          = st.slider("최소 지속시간 (초)",     5,    30,   10,   step=1)

st.sidebar.divider()
all_channels = list(plotting.CHANNEL_ORDER)
default_channels = ["breathing", "pressure", "leakrate", "flowlimit",
                    "spo2", "pulserate", "movement", "sleep_stage"]
selected_channels = st.sidebar.multiselect(
    "표시할 채널", options=all_channels, default=default_channels,
    format_func=lambda c: plotting.CHANNEL_META[c]["title"],
)

st.sidebar.divider()
st.sidebar.caption(
    "⚠️ 한계: CA/OA 구분은 ResMed FOT 채널이 export에 없어 본 검출기는 통합 Apnea만 산출. "
    "AHI 분류는 SleepHQ 이벤트 라벨(timestamp 정확)로 검증합니다."
)

# ─────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────
st.title("🌙 Soom CPAP Quality Analyzer")
st.caption(f"세션 날짜: **{date}** (UTC) · 채널 12종 + Sleep Stage")

session = loader.load_session(DATA_ROOT, date)
if not session:
    st.error("세션을 불러올 수 없습니다.")
    st.stop()

duration_hours = loader.session_duration_hours(session)
events = session.get("events")

# ─────────────────────────────────────────────────────────────────────
# Section 1 — Summary metrics
# ─────────────────────────────────────────────────────────────────────
st.subheader("📋 세션 요약")
m1, m2, m3, m4, m5 = st.columns(5)

ah_mask = events["event_type"].isin(evaluator.APNEA_HYPOPNEA_TYPES) if events is not None else None
ahi_total = (
    (ah_mask.sum() / duration_hours)
    if events is not None and duration_hours and duration_hours > 0 else float("nan")
)
leak_p95 = (
    float(session["leakrate"]["LeakRate_Lpm"].quantile(0.95))
    if "leakrate" in session and not session["leakrate"].empty else float("nan")
)
spo2_lo = (
    float(session["spo2"]["SpO2_pct"].min())
    if "spo2" in session and not session["spo2"].empty else float("nan")
)
press = session.get("pressure")
avg_press = (
    float(press["Pressure_cmH2O"].mean()) if press is not None and not press.empty else float("nan")
)
m1.metric("AHI", f"{ahi_total:.2f}" if pd.notna(ahi_total) else "—", help="SleepHQ 이벤트 / 시간")
m2.metric("Leak 95p", f"{leak_p95:.1f} L/min" if pd.notna(leak_p95) else "—")
m3.metric(
    "Usage",
    f"{int(duration_hours)}h {int(round((duration_hours - int(duration_hours)) * 60)):02d}m"
    if duration_hours else "—",
)
m4.metric("Avg Pressure", f"{avg_press:.1f} cmH₂O" if pd.notna(avg_press) else "—")
m5.metric("Lowest SpO₂", f"{spo2_lo:.1f} %" if pd.notna(spo2_lo) else "—")

# ─────────────────────────────────────────────────────────────────────
# Section 2 — Quality cards (XAI)
# ─────────────────────────────────────────────────────────────────────
st.subheader("🏥 품질 등급 (XAI 카드)")
quality = evaluator.evaluate_quality(session, duration_hours)
cards = quality["cards"]

row1 = st.columns(4)
row2 = st.columns(4)
for i, card in enumerate(cards):
    col = (row1 if i < 4 else row2)[i % 4]
    with col:
        st.markdown(f"#### {card['emoji']} {card['name']}")
        st.markdown(f"**{card['value']} {card['unit']}**")
        st.caption(f"기준: {card['threshold']}")
        st.caption(f"출처: {card['source']}")

overall = quality["overall"]
st.markdown(f"**종합 등급: {overall['emoji']} `{overall['grade'].upper()}`** — {overall['reason']}")

# ─────────────────────────────────────────────────────────────────────
# Section 3 — Detection + agreement
# ─────────────────────────────────────────────────────────────────────
st.subheader("🔍 Apnea / Hypopnea 검출 vs SleepHQ 라벨")

events_aasm = detector.detect_events(session.get("breathing", pd.DataFrame()), detector.PRESETS["aasm"])
events_resmed_params = detector.DetectorParams(
    method="resmed",
    apnea_thr=resmed_apnea_thr, hypop_thr=resmed_hypop_thr,
    min_duration_sec=float(min_dur), baseline_window_sec=100.0,
)
events_resmed = detector.detect_events(session.get("breathing", pd.DataFrame()), events_resmed_params)

c1, c2, c3 = st.columns(3)
c1.metric("AASM 검출", len(events_aasm))
c2.metric("ResMed 검출", len(events_resmed))
sleephq_count = (events["event_type"].isin(evaluator.APNEA_HYPOPNEA_TYPES).sum()
                 if events is not None else 0)
c3.metric("SleepHQ 라벨", int(sleephq_count))

cmp_aasm = evaluator.compare_with_sleephq(events_aasm, events)
cmp_resmed = evaluator.compare_with_sleephq(events_resmed, events)
agree_df = pd.DataFrame([
    {"비교": "AASM vs SleepHQ",   **cmp_aasm},
    {"비교": "ResMed vs SleepHQ", **cmp_resmed},
])
st.dataframe(agree_df, use_container_width=True, hide_index=True)
st.caption("Precision/Recall은 timestamp overlap 단위로 산출됩니다 (각 검출 이벤트가 SleepHQ 이벤트와 시간상 겹치면 TP).")

# ─────────────────────────────────────────────────────────────────────
# Section 4 — Time-series charts (multichannel stacked)
# ─────────────────────────────────────────────────────────────────────
st.subheader("📊 시계열 차트 (시간축 공유)")
st.caption("한 차트에서 zoom/pan 시 모든 채널이 함께 이동합니다.")

overlays: dict[str, pd.DataFrame] = {}
if show_aasm:
    overlays["aasm"] = events_aasm
if show_resmed:
    overlays["resmed"] = events_resmed
if show_sleephq:
    sl = plotting.sleephq_events_as_overlay(events)
    if sl is not None:
        overlays["sleephq"] = sl

st.plotly_chart(
    plotting.plot_multichannel(session, overlays, channels=tuple(selected_channels)),
    use_container_width=True,
)

# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📚 학술 근거 / 출처"):
    st.markdown("""
- **AASM 스코어링**: Berry RB et al. *J Clin Sleep Med* 2012; 8(5): 597–619
- **ResMed AutoSet 알고리즘**: ResMed AirSense 10 AutoSet White Paper
- **FOT (Forced Oscillation Technique)**: Farré R et al. *Eur Respir J* 1998
- **Medicare Adherence**: CMS Decision Memo 2008 (CAG-00093R2) — ≥4 h/night
- **Mask Leak**: ResMed Clinical Guideline — 95th percentile < 24 L/min
- **T90 / 저산소혈증 부담**: Punjabi NM, *Sleep Med* 2009
- **HRV-based Sleep Staging**: Beattie Z et al., *Physiol Meas* 2017

본 도구는 **시연·연구 목적**이며 임상 진단을 대체하지 않습니다.
""")
