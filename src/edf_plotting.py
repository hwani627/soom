"""Plotly figures for the EDF viewer page.

`make_freezoom_figure` renders the whole recording with stride-based
downsampling so each channel is reduced to at most ``MAX_FREEZOOM_POINTS``
samples for stable rendering on Streamlit Cloud.

All callers must pass numpy arrays already loaded via edf_loader.load_signal.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.edf_loader import EdfMeta
from src.hypnogram import HypnogramEpoch, Stage

# Cap per-channel rendered points in free-zoom mode. ~5k points × 7 channels
# stays under Plotly's comfortable ceiling and avoids client-side jank.
MAX_FREEZOOM_POINTS = 5000

# Group → line color (consistent across both freezoom and epoch modes).
CHANNEL_GROUP_COLORS: dict[str, str] = {
    "EEG":   "#0F62FE",
    "EOG":   "#6929C4",
    "EMG":   "#198038",
    "Resp":  "#FF832B",
    "SpO2":  "#EE5396",
    "Other": "#525252",
}

# Sleep stage band colors for the hypnogram strip.
_STAGE_COLORS: dict[Stage, str] = {
    Stage.W:    "#5cb85c",
    Stage.N1:   "#7ab5e0",
    Stage.N2:   "#3a8fc8",
    Stage.N3:   "#c4a13a",
    Stage.REM:  "#c43a3a",
    Stage.MOVE: "#888888",
    Stage.UNK:  "#dddddd",
}


def _stride_downsample(
    sig: np.ndarray, sample_rate: float, max_points: int = MAX_FREEZOOM_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce *sig* to at most *max_points* via fixed-stride sampling.

    Returns ``(t, sig)`` where ``t`` is the time axis in seconds. Cheaper and
    simpler than LTTB; visually adequate for an overview chart of an 8h+ PSG.
    """
    fs = max(sample_rate, 1.0)
    n = len(sig)
    if n <= max_points:
        t = np.arange(n, dtype=np.float64) / fs
        return t, sig
    stride = n // max_points + 1
    return (np.arange(0, n, stride, dtype=np.float64) / fs,
            sig[::stride])


def make_freezoom_figure(
    meta: EdfMeta,
    signals: dict[int, np.ndarray],
    hypno: list[HypnogramEpoch] | None,
    channels: Iterable[int],
) -> go.Figure:
    """Stacked-subplot view of the whole recording.

    Each channel is stride-downsampled to at most ``MAX_FREEZOOM_POINTS``
    points before plotting so the page stays responsive even with 8h+ PSG
    data. For full-fidelity inspection use ``make_epoch_figure`` instead.

    Args:
        meta: Result of `edf_loader.load_meta`.
        signals: Mapping ``ch_idx -> numpy array`` (one entry per visible channel).
        hypno: Optional list of HypnogramEpoch — drawn as a colored band below.
        channels: Channel indices to plot, in display order (top to bottom).
    """
    chans = [c for c in channels if c in signals]
    n_chan_rows = len(chans)
    if n_chan_rows == 0:
        return go.Figure()

    has_hypno = bool(hypno)
    total_rows = n_chan_rows + (1 if has_hypno else 0)
    row_heights = [1.0] * n_chan_rows + ([0.25] if has_hypno else [])
    titles = [meta.channels[c].label for c in chans] + (
        ["Sleep stage (Hypnogram)"] if has_hypno else []
    )

    fig = make_subplots(
        rows=total_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=row_heights,
        subplot_titles=titles,
    )

    for row_idx, ch in enumerate(chans, start=1):
        info = meta.channels[ch]
        t, y = _stride_downsample(signals[ch], info.sample_rate)
        fig.add_trace(
            go.Scatter(
                x=t, y=y, mode="lines",
                line=dict(width=1.0,
                          color=CHANNEL_GROUP_COLORS.get(info.group,
                                                         CHANNEL_GROUP_COLORS["Other"])),
                name=info.label, showlegend=False,
            ),
            row=row_idx, col=1,
        )
        fig.update_yaxes(title_text=info.physical_dim or "", row=row_idx, col=1)

    if has_hypno:
        _add_hypnogram_band(fig, hypno, row=total_rows)

    fig.update_xaxes(title_text="Time (s)", row=total_rows, col=1)
    fig.update_layout(
        height=180 * n_chan_rows + (60 if has_hypno else 0) + 80,
        margin=dict(l=55, r=20, t=40, b=40),
        hovermode="x unified",
    )
    return fig


def _add_hypnogram_band(
    fig: go.Figure, epochs: list[HypnogramEpoch], row: int
) -> None:
    """Render the hypnogram as horizontal colored rectangles in the given row."""
    # Plotly axis ref naming: row 1 → 'x'/'y', row N>1 → 'xN'/'yN'
    xref = "x" if row == 1 else f"x{row}"
    yref_domain = "y domain" if row == 1 else f"y{row} domain"
    for e in epochs:
        fig.add_shape(
            type="rect", xref=xref, yref=yref_domain,
            x0=e.start_sec, x1=e.start_sec + e.duration_sec,
            y0=0, y1=1,
            fillcolor=_STAGE_COLORS.get(e.stage, _STAGE_COLORS[Stage.UNK]),
            line=dict(width=0), layer="below",
        )
    fig.update_yaxes(visible=False, row=row, col=1)


def make_epoch_figure(
    meta: EdfMeta,
    signals: dict[int, np.ndarray],
    hypno: list[HypnogramEpoch] | None,
    channels: Iterable[int],
    epoch_idx: int,
    epoch_sec: float = 30.0,
) -> go.Figure:
    """30-second epoch view (PSG clinical standard).

    No downsampling — one epoch holds at most ~10k samples per channel which
    Plotly renders at full fidelity. A minimap row at the bottom shows the
    full-night hypnogram plus a vertical marker at the current epoch.
    """
    chans = [c for c in channels if c in signals]
    if not chans:
        return go.Figure()

    t0 = epoch_idx * epoch_sec
    t1 = t0 + epoch_sec

    has_minimap = bool(hypno)
    total_rows = len(chans) + (1 if has_minimap else 0)
    row_heights = [1.0] * len(chans) + ([0.35] if has_minimap else [])
    titles = [meta.channels[c].label for c in chans] + (
        ["Hypnogram (current epoch marked)"] if has_minimap else []
    )

    fig = make_subplots(
        rows=total_rows, cols=1,
        shared_xaxes=False,  # minimap uses full-night x-axis, channels use epoch slice
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=titles,
    )

    for row_idx, ch in enumerate(chans, start=1):
        info = meta.channels[ch]
        sig = signals[ch]
        fs = max(info.sample_rate, 1.0)
        a = int(t0 * fs)
        b = int(min(len(sig), t1 * fs))
        t = np.arange(a, b, dtype=np.float64) / fs
        fig.add_trace(
            go.Scatter(
                x=t, y=sig[a:b], mode="lines",
                line=dict(width=1.0,
                          color=CHANNEL_GROUP_COLORS.get(info.group,
                                                         CHANNEL_GROUP_COLORS["Other"])),
                name=info.label, showlegend=False,
            ),
            row=row_idx, col=1,
        )
        fig.update_yaxes(title_text=info.physical_dim or "", row=row_idx, col=1)
        fig.update_xaxes(range=[t0, t1], row=row_idx, col=1)

    if has_minimap:
        _add_hypnogram_minimap(fig, hypno, current_t=t0,
                               total_dur=meta.duration_sec, row=total_rows)

    fig.update_xaxes(title_text="Time (s)", row=total_rows, col=1)
    fig.update_layout(
        height=180 * len(chans) + (80 if has_minimap else 0) + 80,
        margin=dict(l=55, r=20, t=40, b=40),
        hovermode="x unified",
    )
    return fig


def _add_hypnogram_minimap(
    fig: go.Figure, epochs: list[HypnogramEpoch],
    current_t: float, total_dur: float, row: int,
) -> None:
    """Render full-night hypnogram band + a vertical marker at current_t."""
    xref = "x" if row == 1 else f"x{row}"
    yref = "y domain" if row == 1 else f"y{row} domain"
    for e in epochs:
        fig.add_shape(
            type="rect", xref=xref, yref=yref,
            x0=e.start_sec, x1=e.start_sec + e.duration_sec,
            y0=0, y1=1,
            fillcolor=_STAGE_COLORS.get(e.stage, _STAGE_COLORS[Stage.UNK]),
            line=dict(width=0), layer="below",
        )
    # Current-epoch marker
    fig.add_shape(
        type="line", xref=xref, yref=yref,
        x0=current_t, x1=current_t, y0=0, y1=1,
        line=dict(color="black", width=2),
    )
    fig.update_yaxes(visible=False, row=row, col=1)
    fig.update_xaxes(range=[0, total_dur], row=row, col=1)
