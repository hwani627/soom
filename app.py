"""Soom CPAP Quality Analyzer — Streamlit demo app.

Run locally:    streamlit run app.py
Deploy:         Streamlit Community Cloud (https://share.streamlit.io)
"""
from __future__ import annotations

import io
import tempfile
import zipfile
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
# Sidebar — data source + algorithm controls
# ─────────────────────────────────────────────────────────────────────
st.sidebar.title("🌙 Soom Analyzer")
st.sidebar.caption("PSG-Level CPAP Demo")

source_mode = st.sidebar.radio(
    "데이터 소스",
    options=["번들 데모", "사용자 업로드"],
    index=0,
)

available_dates = loader.list_available_dates(DATA_ROOT)

session_dir: Path | None = None
date_label: str = "—"

if source_mode == "번들 데모":
    if not available_dates:
        st.sidebar.error(f"번들 데이터가 없습니다: {DATA_ROOT}")
    else:
        chosen = st.sidebar.selectbox(
            "날짜 선택", options=available_dates, index=len(available_dates) - 1,
            format_func=lambda d: f"{d[:4]}-{d[4:6]}-{d[6:8]}",
        )
        session_dir = DATA_ROOT / chosen
        date_label = f"{chosen[:4]}-{chosen[4:6]}-{chosen[6:8]}"
else:
    uploaded = st.sidebar.file_uploader(
        "ZIP 업로드 (YYYYMMDD/*.csv 구조)", type=["zip"], accept_multiple_files=False,
    )
    if uploaded is not None:
        tmp_dir = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(io.BytesIO(uploaded.read())) as zf:
            zf.extractall(tmp_dir)
        # Find first 8-digit folder inside
        candidates = [p for p in tmp_dir.rglob("*") if p.is_dir() and p.name.isdigit() and len(p.name) == 8]
        if candidates:
            session_dir = candidates[0]
            date_label = session_dir.name
        else:
            st.sidebar.error("ZIP 안에 YYYYMMDD/ 폴더를 찾을 수 없습니다.")

st.sidebar.divider()
st.sidebar.markdown("**검출 알고리즘**")
show_aasm = st.sidebar.checkbox("AASM (≥90% / ≥30%)", value=True)
show_resmed = st.sidebar.checkbox("ResMed (<25% / <50%)", value=True)
show_sleephq = st.sidebar.checkbox("SleepHQ 라벨 (placeholder)", value=True)

with st.sidebar.expander("ResMed 임계값 슬라이더"):
    resmed_apnea_thr = st.slider("Apnea 임계 (ratio)", 0.05, 0.50, 0.25, step=0.01)
    resmed_hypop_thr = st.slider("Hypopnea 임계 (ratio)", 0.30, 0.90, 0.50, step=0.01)
    min_dur = st.slider("최소 지속시간 (초)", 5, 30, 10, step=1)

st.sidebar.divider()
st.sidebar.caption(
    "⚠️ 한계: CA(Clear Airway) vs OA(Obstructive) 구분은 FOT 채널이 export CSV에 "
    "없어 1차 버전에서 불가합니다. AHI 분류는 SleepHQ Summary 값을 사용합니다."
)

# ─────────────────────────────────────────────────────────────────────
# Main — header
# ─────────────────────────────────────────────────────────────────────
st.title("🌙 Soom CPAP Quality Analyzer")
st.caption(f"세션 날짜: **{date_label}** · 데모 v0.1.0")

if session_dir is None:
    st.info("좌측에서 날짜를 선택하거나 ZIP을 업로드하세요.")
    st.stop()

# Load
session = loader.load_session(session_dir)
if not session:
    st.error("세션을 불러올 수 없습니다.")
    st.stop()

duration_hours = loader.session_duration_hours(session)

# ─────────────────────────────────────────────────────────────────────
# Section 1 — Summary metrics (4 cards)
# ─────────────────────────────────────────────────────────────────────
st.subheader("📋 세션 요약")
m1, m2, m3, m4 = st.columns(4)
ahi_sum = session.get("ahi_summary")
ahi_total = (
    float(ahi_sum.loc[ahi_sum["Metric"] == "Total Events", "Value"].iloc[0])
    if ahi_sum is not None and not ahi_sum.empty and (ahi_sum["Metric"] == "Total Events").any()
    else float("nan")
)
leak_p95 = (
    float(session["leakrate"]["LeakRate_Lpm"].quantile(0.95))
    if "leakrate" in session else float("nan")
)
m1.metric("AHI", f"{ahi_total:.2f}" if pd.notna(ahi_total) else "—", help="Total Events / hour (SleepHQ)")
m2.metric("Leak 95p (L/min)", f"{leak_p95:.1f}" if pd.notna(leak_p95) else "—")
m3.metric(
    "Usage",
    f"{int(duration_hours)}h {int(round((duration_hours - int(duration_hours)) * 60)):02d}m"
    if duration_hours else "—",
)
press = session.get("pressure")
avg_press = (
    float(press["Pressure_cmH2O"].mean()) if press is not None and not press.empty else float("nan")
)
m4.metric("Avg Pressure (cmH₂O)", f"{avg_press:.2f}" if pd.notna(avg_press) else "—")

# ─────────────────────────────────────────────────────────────────────
# Section 2 — Quality cards (XAI)
# ─────────────────────────────────────────────────────────────────────
st.subheader("🏥 품질 등급 (XAI 카드)")
quality = evaluator.evaluate_quality(session, duration_hours)
cols = st.columns(4)
for col, card in zip(cols, quality["cards"]):
    with col:
        st.markdown(f"#### {card['emoji']} {card['name']}")
        st.markdown(f"**{card['value']} {card['unit']}**")
        st.caption(f"기준: {card['threshold']}")
        st.caption(f"출처: {card['source']}")

overall = quality["overall"]
st.markdown(
    f"**종합 등급: {overall['emoji']} `{overall['grade'].upper()}`** — {overall['reason']}"
)

# ─────────────────────────────────────────────────────────────────────
# Section 3 — Detection
# ─────────────────────────────────────────────────────────────────────
st.subheader("🔍 Apnea / Hypopnea 검출")

events_aasm = detector.detect_events(session.get("breathing", pd.DataFrame()), detector.PRESETS["aasm"])
events_resmed_params = detector.DetectorParams(
    method="resmed",
    apnea_thr=resmed_apnea_thr,
    hypop_thr=resmed_hypop_thr,
    min_duration_sec=float(min_dur),
    baseline_window_sec=100.0,
)
events_resmed = detector.detect_events(session.get("breathing", pd.DataFrame()), events_resmed_params)

events_sleephq = plotting.plot_synthetic_sleephq_from_summary(
    duration_hours, session.get("ahi_summary"), session.get("breathing"),
)

c1, c2, c3 = st.columns(3)
c1.metric("AASM 검출", len(events_aasm))
c2.metric("ResMed 검출", len(events_resmed))
c3.metric("SleepHQ 카운트", len(events_sleephq) if events_sleephq is not None else "—")

# Agreement (count-based, not timestamp-accurate)
agree_aasm = evaluator.compare_with_sleephq(events_aasm, ahi_sum, duration_hours)
agree_resmed = evaluator.compare_with_sleephq(events_resmed, ahi_sum, duration_hours)
agree_df = pd.DataFrame([
    {"비교": "AASM vs SleepHQ", **agree_aasm},
    {"비교": "ResMed vs SleepHQ", **agree_resmed},
])
st.dataframe(agree_df, use_container_width=True, hide_index=True)
st.caption(
    "ℹ️ 일치율은 **카운트 단위**입니다. SleepHQ 라벨의 timestamp 시계열이 export CSV에 "
    "포함되지 않아 timestamp 단위 매칭은 1차 버전 범위 외입니다."
)

# ─────────────────────────────────────────────────────────────────────
# Section 4 — Time-series charts
# ─────────────────────────────────────────────────────────────────────
st.subheader("📊 시계열 차트")

overlays: dict[str, pd.DataFrame] = {}
if show_aasm:
    overlays["aasm"] = events_aasm
if show_resmed:
    overlays["resmed"] = events_resmed
if show_sleephq and events_sleephq is not None:
    overlays["sleephq"] = events_sleephq

st.caption("4개 채널이 시간축을 공유합니다. 한 차트에서 zoom/pan 시 모두 함께 이동합니다.")
st.plotly_chart(plotting.plot_multichannel(session, overlays), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📚 학술 근거 / 출처"):
    st.markdown("""
- **AASM 스코어링**: Berry RB et al. *J Clin Sleep Med* 2012; 8(5): 597–619
- **ResMed AutoSet 알고리즘**: ResMed AirSense 10 AutoSet White Paper
- **FOT (Forced Oscillation Technique)**: Farré R et al. *Eur Respir J* 1998
- **Medicare Adherence**: CMS Decision Memo 2008 (CAG-00093R2) — ≥4 h/night × ≥70% nights
- **Mask Leak**: ResMed Clinical Guideline — 95th percentile < 24 L/min

본 도구는 **시연·연구 목적**이며 임상 진단을 대체하지 않습니다.
""")
