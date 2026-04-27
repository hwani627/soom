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

# Compact typography for clinical readability
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
      h1 { font-size: 1.4rem !important; margin-bottom: 0.3rem; }
      h2 { font-size: 1.05rem !important; margin-top: 0.8rem; margin-bottom: 0.3rem; }
      h3 { font-size: 0.95rem !important; }
      .stMarkdown p, .stCaption, label, .stRadio label, .stCheckbox label {
          font-size: 0.85rem !important;
      }
      [data-testid="stCaptionContainer"] { font-size: 0.78rem !important; }
      .stDataFrame { font-size: 0.82rem; }
      .stDataFrame tbody td, .stDataFrame thead th { padding: 4px 8px !important; }
      [data-testid="stSidebar"] .stMarkdown,
      [data-testid="stSidebar"] label { font-size: 0.82rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="📥 데이터 로딩 중...")
def _load_session_cached(date: str) -> dict:
    return loader.load_session(DATA_ROOT, date)


@st.cache_data(show_spinner=False)
def _detect_aasm_cached(date: str) -> pd.DataFrame:
    s = _load_session_cached(date)
    return detector.detect_events(s.get("breathing", pd.DataFrame()), detector.PRESETS["aasm"])


@st.cache_data(show_spinner=False)
def _detect_resmed_cached(date: str, apnea_thr: float, hypop_thr: float, min_dur: int) -> pd.DataFrame:
    s = _load_session_cached(date)
    params = detector.DetectorParams(
        method="resmed", apnea_thr=apnea_thr, hypop_thr=hypop_thr,
        min_duration_sec=float(min_dur), baseline_window_sec=100.0,
    )
    return detector.detect_events(s.get("breathing", pd.DataFrame()), params)

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

session = _load_session_cached(date)
if not session:
    st.error("세션을 불러올 수 없습니다.")
    st.stop()

duration_hours = loader.session_duration_hours(session)
events = session.get("events")

# ─────────────────────────────────────────────────────────────────────
# Section 1 — Summary table
# ─────────────────────────────────────────────────────────────────────
st.subheader("📋 세션 요약")

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


def _fmt(v, fmt: str) -> str:
    return fmt.format(v) if pd.notna(v) else "—"


usage_str = (
    f"{int(duration_hours)}h {int(round((duration_hours - int(duration_hours)) * 60)):02d}m"
    if duration_hours else "—"
)
summary_df = pd.DataFrame([{
    "AHI (/h)":          _fmt(ahi_total, "{:.2f}"),
    "Leak 95p (L/min)":  _fmt(leak_p95,  "{:.1f}"),
    "Usage":             usage_str,
    "Avg Pressure (cmH₂O)": _fmt(avg_press, "{:.1f}"),
    "Lowest SpO₂ (%)":   _fmt(spo2_lo, "{:.1f}"),
}])
st.dataframe(summary_df, hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# Section 2 — Quality cards as table
# ─────────────────────────────────────────────────────────────────────
st.subheader("🏥 품질 등급 (XAI 카드)")
quality = evaluator.evaluate_quality(session, duration_hours)
cards = quality["cards"]

cards_df = pd.DataFrame([
    {
        "지표": c["name"],
        "측정값": f"{c['value']}{(' ' + c['unit']) if c['unit'] else ''}",
        "등급": c["emoji"],
        "기준": c["threshold"],
        "출처": c["source"],
    }
    for c in cards
])
st.dataframe(cards_df, hide_index=True, use_container_width=True)

overall = quality["overall"]
st.markdown(
    f"**종합 등급: {overall['emoji']} `{overall['grade'].upper()}`** &nbsp;·&nbsp; "
    f"<span style='font-size:0.85rem;color:#666;'>{overall['reason']}</span>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────
# Section 3 — Detection + agreement
# ─────────────────────────────────────────────────────────────────────
st.subheader("🔍 Apnea / Hypopnea 검출 vs SleepHQ 라벨")

events_aasm = _detect_aasm_cached(date)
events_resmed = _detect_resmed_cached(date, resmed_apnea_thr, resmed_hypop_thr, min_dur)

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
