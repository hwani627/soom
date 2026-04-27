"""Plotly figures for time-series + event overlays."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

CHANNEL_META = {
    "breathing": {"col": "Breathing_Lpm", "title": "Breathing (L/min)", "y": "Flow (L/min)"},
    "pressure":  {"col": "Pressure_cmH2O", "title": "Pressure (cmH₂O)", "y": "cmH₂O"},
    "leakrate":  {"col": "LeakRate_Lpm", "title": "Leak Rate (L/min)", "y": "L/min"},
    "flowlimit": {"col": "FlowLimit", "title": "Flow Limit", "y": "FL index"},
}

OVERLAY_COLORS = {
    "aasm":    "rgba(255,0,0,0.18)",     # red
    "resmed":  "rgba(0,0,255,0.18)",     # blue
    "sleephq": "rgba(0,160,0,0.18)",     # green
}


def _add_event_shapes(fig: go.Figure, events: pd.DataFrame, color: str, label: str) -> None:
    if events is None or events.empty:
        return
    for _, row in events.iterrows():
        fig.add_vrect(
            x0=row["start_ts"], x1=row["end_ts"],
            fillcolor=color, opacity=0.5, line_width=0,
            annotation_text=row["type"][:1].upper(), annotation_position="top left",
            annotation=dict(font_size=9),
        )
    # Legend dummy
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color=color.replace("0.18", "0.6")),
        name=label, showlegend=True,
    ))


def plot_timeseries(
    df: Optional[pd.DataFrame],
    channel: str,
    overlays: Optional[dict[str, pd.DataFrame]] = None,
) -> go.Figure:
    meta = CHANNEL_META[channel]
    fig = go.Figure()
    if df is not None and not df.empty and meta["col"] in df.columns:
        # WebGL for high-density Breathing
        trace_cls = go.Scattergl if channel == "breathing" else go.Scatter
        fig.add_trace(trace_cls(
            x=df["Timestamp_ET"], y=df[meta["col"]],
            mode="lines", line=dict(width=1, color="#0F62FE"),
            name=meta["title"], showlegend=False,
        ))
    if overlays:
        for method, evs in overlays.items():
            if method not in OVERLAY_COLORS:
                continue
            _add_event_shapes(fig, evs, OVERLAY_COLORS[method], label=method.upper())
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
