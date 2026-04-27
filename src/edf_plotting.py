"""Plotly figures for the EDF viewer page.

`make_freezoom_figure` wraps plotly_resampler.FigureResampler so that the
plot dynamically downsamples (LTTB) on zoom — see Steinarsson 2013 and
Van Der Donckt et al., SoftwareX 2022.

All callers must pass numpy arrays already loaded via edf_loader.load_signal.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler

from src.edf_loader import EdfMeta
from src.hypnogram import HypnogramEpoch, Stage

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


def make_freezoom_figure(
    meta: EdfMeta,
    signals: dict[int, np.ndarray],
    hypno: list[HypnogramEpoch] | None,
    channels: Iterable[int],
) -> go.Figure:
    """Stacked-subplot view of the whole recording, with dynamic LTTB.

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

    base = make_subplots(
        rows=total_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=row_heights,
        subplot_titles=titles,
    )
    fig = FigureResampler(base, default_n_shown_samples=4000)

    for row_idx, ch in enumerate(chans, start=1):
        info = meta.channels[ch]
        sig = signals[ch]
        t = np.arange(len(sig), dtype=np.float64) / max(info.sample_rate, 1.0)
        fig.add_trace(
            go.Scatter(
                mode="lines",
                line=dict(width=1.0,
                          color=CHANNEL_GROUP_COLORS.get(info.group,
                                                         CHANNEL_GROUP_COLORS["Other"])),
                name=info.label, showlegend=False,
            ),
            hf_x=t, hf_y=sig,
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
