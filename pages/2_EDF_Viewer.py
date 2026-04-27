"""EDF Viewer (PSG) — independent page in the Soom Streamlit app.

Loads `*.edf` files from the repo's CAPA_Data folder, lets the user pick
channels and a viewing mode (free zoom / 30-second epoch), and overlays
the paired Hypnogram annotation when available.

Spec: docs/specs/2026-04-27-edf-viewer-design.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import edf_loader, edf_plotting, hypnogram

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "CAPA_Data"

st.set_page_config(
    page_title="EDF Viewer · Soom",
    page_icon="🧠",
    layout="wide",
)

# Same compact CSS as app.py — Streamlit multipage applies CSS per page.
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
      h1 { font-size: 1.4rem !important; margin-bottom: 0.3rem; }
      h2 { font-size: 1.05rem !important; margin-top: 0.8rem; margin-bottom: 0.3rem; }
      h3 { font-size: 0.95rem !important; }
      .stMarkdown p, .stCaption, label { font-size: 0.85rem !important; }
      [data-testid="stCaptionContainer"] { font-size: 0.78rem !important; }
      [data-testid="stSidebar"] .stMarkdown,
      [data-testid="stSidebar"] label { font-size: 0.82rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def _list_files_cached(root_str: str) -> list[str]:
    return [str(p) for p in edf_loader.list_edf_files(Path(root_str))]


@st.cache_data(show_spinner="📥 EDF 헤더 읽는 중...")
def _load_meta_cached(path_str: str):
    return edf_loader.load_meta(Path(path_str))


@st.cache_data(show_spinner=False, max_entries=10)
def _load_signal_cached(path_str: str, ch_idx: int) -> np.ndarray:
    return edf_loader.load_signal(Path(path_str), ch_idx)


@st.cache_data(show_spinner=False)
def _load_hypnogram_cached(path_str: str):
    return hypnogram.load_hypnogram(Path(path_str))


# ─────────────────────────────────────────────────────────────────────
# Sidebar — file + channel + mode selectors
# ─────────────────────────────────────────────────────────────────────
st.sidebar.title("🧠 EDF Viewer")
st.sidebar.caption("PSG 표준 EDF/EDF+ · v1")

paths = _list_files_cached(str(DATA_ROOT))
if not paths:
    st.sidebar.error(f"EDF 파일을 찾을 수 없습니다: {DATA_ROOT}")
    st.info("`CAPA_Data/` 폴더에 `*.edf` 파일을 넣어주세요. (`*-Hypnogram.edf`는 자동 페어링되며 목록에 표시되지 않습니다.)")
    st.stop()

selected_path_str = st.sidebar.selectbox(
    "EDF 파일",
    options=paths,
    format_func=lambda p: Path(p).name,
)

try:
    meta = _load_meta_cached(selected_path_str)
except (OSError, ValueError, RuntimeError) as e:
    st.error(f"EDF 헤더를 읽을 수 없습니다: {e}")
    st.stop()

# Spec §7 — large-file guard (warning only in v1; no slicing)
if meta.duration_sec > 86400 or len(meta.channels) > 50:
    st.warning(
        f"⚠️ 대용량 EDF: duration={meta.duration_sec / 3600:.1f}h, "
        f"channels={len(meta.channels)}. 렌더링이 느릴 수 있습니다."
    )

# Channel group toggles
all_groups = sorted({ch.group for ch in meta.channels})
st.sidebar.markdown("**채널 그룹 (Channel Groups)**")
group_state: dict[str, bool] = {}
for g in all_groups:
    group_state[g] = st.sidebar.checkbox(g, value=(g != "Other"), key=f"grp_{g}")

# Filtered individual-channel multiselect
all_ch_indices = [i for i, ch in enumerate(meta.channels) if group_state.get(ch.group, False)]
selected_ch_indices = st.sidebar.multiselect(
    "표시할 채널",
    options=all_ch_indices,
    default=all_ch_indices,
    format_func=lambda i: f"{meta.channels[i].label} ({meta.channels[i].group}, {meta.channels[i].sample_rate:.0f} Hz)",
)

# Mode toggle
st.sidebar.divider()
mode = st.sidebar.radio(
    "보기 모드",
    options=["자유 줌 (Free Zoom)", "30s 에포크 (Epoch)"],
    index=0,
)

# Epoch slider (only in epoch mode)
if mode.startswith("30s"):
    total_epochs = max(1, int(meta.duration_sec // 30))
    epoch_idx = st.sidebar.slider(
        "에포크 (Epoch)",
        min_value=0, max_value=total_epochs - 1, value=0, step=1,
        format="%d",
    )
else:
    epoch_idx = 0

st.sidebar.divider()
st.sidebar.caption("ⓘ Hypnogram 파일이 같은 폴더에 있으면 자동 페어링됩니다.")


# ─────────────────────────────────────────────────────────────────────
# Header panel
# ─────────────────────────────────────────────────────────────────────
st.title("🧠 EDF Viewer (PSG)")
header_df = pd.DataFrame([{
    "파일": Path(meta.path).name,
    "환자 ID": meta.subject_id,
    "시작": meta.start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
    "길이 (시간)": f"{meta.duration_sec / 3600:.2f}",
    "채널 수": len(meta.channels),
}])
st.dataframe(header_df, hide_index=True, use_container_width=True)

# Channel breakdown table
ch_df = pd.DataFrame([
    {"#": i, "라벨": ch.label, "그룹": ch.group,
     "샘플레이트 (Hz)": ch.sample_rate, "단위": ch.physical_dim,
     "샘플 수": ch.n_samples}
    for i, ch in enumerate(meta.channels)
])
with st.expander("채널 상세 (Channel details)"):
    st.dataframe(ch_df, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# Hypnogram pairing
# ─────────────────────────────────────────────────────────────────────
hyp_path = edf_loader.pair_hypnogram(Path(selected_path_str))
hypno = None
if hyp_path is not None:
    try:
        hypno = _load_hypnogram_cached(str(hyp_path))
    except (OSError, ValueError, RuntimeError) as e:
        st.warning(f"Hypnogram 파일을 읽을 수 없어 오버레이를 비활성화합니다: {e}")
        hypno = None
    else:
        st.caption(f"✅ Hypnogram 페어링: `{hyp_path.name}` ({len(hypno)} epochs)")
else:
    st.caption("ℹ️ Hypnogram 파일이 없어 sleep stage 오버레이가 비활성화되었습니다.")


# ─────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────
st.subheader("📊 시계열 (Time-series)")

if not selected_ch_indices:
    st.warning("채널을 1개 이상 선택하세요.")
    st.stop()

# Lazy load only the selected channels
signals = {i: _load_signal_cached(selected_path_str, i)
           for i in selected_ch_indices}

if mode.startswith("자유"):
    fig = edf_plotting.make_freezoom_figure(
        meta, signals, hypno=hypno, channels=selected_ch_indices,
    )
else:
    fig = edf_plotting.make_epoch_figure(
        meta, signals, hypno=hypno, channels=selected_ch_indices,
        epoch_idx=epoch_idx, epoch_sec=30.0,
    )

st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "본 도구는 **시연·연구 목적**의 EDF/EDF+ 뷰어이며 임상 진단을 대체하지 않습니다. "
    "자유 줌 모드는 초기 렌더 시 LTTB 다운샘플링을 적용합니다(plotly-resampler, Steinarsson 2013). 정밀 검토는 30s 에포크 모드를 사용하세요."
)
