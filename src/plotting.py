"""Plotly figures for time-series + event overlays."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CHANNEL_META = {
    "breathing": {"col": "Breathing_Lpm", "title": "Breathing", "y": "Flow (L/min)", "color": "#0F62FE"},
    "pressure":  {"col": "Pressure_cmH2O", "title": "Pressure", "y": "cmH₂O",       "color": "#DA1E28"},
    "leakrate":  {"col": "LeakRate_Lpm", "title": "Leak Rate",  "y": "L/min",       "color": "#FF832B"},
    "flowlimit": {"col": "FlowLimit",    "title": "Flow Limit", "y": "FL index",    "color": "#198038"},
}

CHANNEL_ORDER = ("breathing", "pressure", "leakrate", "flowlimit")

OVERLAY_COLORS = {
    "aasm":    "rgba(255,0,0,0.18)",     # red
    "resmed":  "rgba(0,0,255,0.18)",     # blue
    "sleephq": "rgba(0,160,0,0.18)",     # green
}


def _build_event_shapes(events: pd.DataFrame, color: str) -> list[dict]:
    if events is None or events.empty:
        return []
    return [
        dict(
            type="rect", xref="x", yref="paper",
            x0=row["start_ts"], x1=row["end_ts"], y0=0, y1=1,
            fillcolor=color, opacity=0.5, line=dict(width=0), layer="below",
        )
        for _, row in events.iterrows()
    ]


def _legend_dummy(color: str, label: str) -> go.Scatter:
    return go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color=color.replace("0.18", "0.6")),
        name=label, showlegend=True,
    )


def _downsample(df: pd.DataFrame, max_points: int = 8000) -> pd.DataFrame:
    """Stride-based downsampling for fast SVG rendering on dense series (e.g. Breathing)."""
    if df is None or len(df) <= max_points:
        return df
    stride = len(df) // max_points + 1
    return df.iloc[::stride].reset_index(drop=True)


def plot_timeseries(
    df: Optional[pd.DataFrame],
    channel: str,
    overlays: Optional[dict[str, pd.DataFrame]] = None,
) -> go.Figure:
    """Single-channel chart (kept for backward compatibility / individual exports)."""
    meta = CHANNEL_META[channel]
    fig = go.Figure()
    if df is not None and not df.empty and meta["col"] in df.columns:
        plot_df = _downsample(df, max_points=8000)
        fig.add_trace(go.Scatter(
            x=plot_df["Timestamp_ET"], y=plot_df[meta["col"]],
            mode="lines", line=dict(width=1, color=meta["color"]),
            name=meta["title"], showlegend=False,
        ))
    if overlays:
        all_shapes: list[dict] = []
        for method, evs in overlays.items():
            if method not in OVERLAY_COLORS:
                continue
            all_shapes.extend(_build_event_shapes(evs, OVERLAY_COLORS[method]))
            fig.add_trace(_legend_dummy(OVERLAY_COLORS[method], method.upper()))
        if all_shapes:
            fig.update_layout(shapes=all_shapes)
    fig.update_layout(
        title=meta["title"],
        xaxis_title="Time",
        yaxis_title=meta["y"],
        height=380,
        margin=dict(l=40, r=20, t=40, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_multichannel(
    session: dict,
    overlays: Optional[dict[str, pd.DataFrame]] = None,
    height_per_row: int = 220,
) -> go.Figure:
    """Stacked 4-row chart: Breathing / Pressure / LeakRate / FlowLimit with shared x-axis.

    Each row uses its own color. Event overlays (Apnea/Hypopnea/SleepHQ) are applied
    to every row so a vertical band crosses all four panels at the same time window.
    """
    titles = [f"{CHANNEL_META[c]['title']}  ({CHANNEL_META[c]['y']})" for c in CHANNEL_ORDER]
    fig = make_subplots(
        rows=len(CHANNEL_ORDER), cols=1,
        shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=titles,
    )

    for i, ch in enumerate(CHANNEL_ORDER, start=1):
        meta = CHANNEL_META[ch]
        df = session.get(ch)
        if df is not None and not df.empty and meta["col"] in df.columns:
            plot_df = _downsample(df, max_points=8000)
            fig.add_trace(
                go.Scatter(
                    x=plot_df["Timestamp_ET"], y=plot_df[meta["col"]],
                    mode="lines", line=dict(width=1, color=meta["color"]),
                    name=meta["title"], showlegend=False, hovertemplate=f"%{{x}}<br>%{{y}} {meta['y']}<extra></extra>",
                ),
                row=i, col=1,
            )
        fig.update_yaxes(title_text=meta["y"], row=i, col=1)

    # Event overlays: one rect per row per event
    if overlays:
        all_shapes: list[dict] = []
        for method, evs in overlays.items():
            if method not in OVERLAY_COLORS or evs is None or evs.empty:
                continue
            for r in range(1, len(CHANNEL_ORDER) + 1):
                xref = "x" if r == 1 else f"x{r}"
                yref = "y domain" if r == 1 else f"y{r} domain"
                for _, row in evs.iterrows():
                    all_shapes.append(dict(
                        type="rect", xref=xref, yref=yref,
                        x0=row["start_ts"], x1=row["end_ts"], y0=0, y1=1,
                        fillcolor=OVERLAY_COLORS[method], opacity=0.5,
                        line=dict(width=0), layer="below",
                    ))
            fig.add_trace(_legend_dummy(OVERLAY_COLORS[method], method.upper()), row=1, col=1)
        if all_shapes:
            fig.update_layout(shapes=all_shapes)

    fig.update_xaxes(title_text="Time", row=len(CHANNEL_ORDER), col=1)
    fig.update_layout(
        height=height_per_row * len(CHANNEL_ORDER) + 60,
        margin=dict(l=50, r=20, t=40, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_synthetic_sleephq_from_summary(
    duration_hours: Optional[float],
    ahi_summary: Optional[pd.DataFrame],
    breathing: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """SleepHQ label time-series is not exported in CSV. We synthesize evenly-spaced
    placeholder marks for visual count comparison only (NOT timestamp-accurate).
    """
    if (duration_hours is None or duration_hours <= 0
            or ahi_summary is None or ahi_summary.empty
            or breathing is None or breathing.empty):
        return None
    row = ahi_summary[ahi_summary["Metric"] == "Total Events"] if "Metric" in ahi_summary.columns else None
    if row is None or row.empty:
        return None
    n = int(round(float(row["Value"].iloc[0]) * duration_hours))
    if n <= 0:
        return None
    t0 = breathing["Timestamp_ET"].iloc[0]
    t1 = breathing["Timestamp_ET"].iloc[-1]
    marks = pd.date_range(t0, t1, periods=n + 2)[1:-1]
    span = pd.Timedelta(seconds=15)
    return pd.DataFrame({
        "start_ts": marks,
        "end_ts": marks + span,
        "type": "sleephq",
        "duration_sec": [span.total_seconds()] * n,
        "method": "sleephq",
    })
