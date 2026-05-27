"""utils/visualizer.py — Charts and plots for the Streamlit UI."""

from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def plot_confidence_gauge(confidence: float, label: str):
    """Gauge chart showing prediction confidence."""
    color = "#f5576c" if label == "Dog" else "#4facfe"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [50, 100], "ticksuffix": "%"},
            "bar": {"color": color},
            "bgcolor": "#1e1e2e",
            "bordercolor": "#2a2a3e",
            "steps": [
                {"range": [50, 70], "color": "#2a2a3e"},
                {"range": [70, 90], "color": "#333350"},
                {"range": [90, 100], "color": "#3d3d60"},
            ],
        },
        title={"text": f"{label} Confidence", "font": {"size": 16}},
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="white"
    )
    return fig


def plot_prediction_history(history: list[dict]):
    """Bar chart of Dog vs Cat counts in session history."""
    if not history:
        return None

    df = pd.DataFrame(history)
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["Label", "Count"]

    color_map = {"Dog": "#f5576c", "Cat": "#4facfe"}
    fig = px.bar(
        counts, x="Label", y="Count",
        color="Label",
        color_discrete_map=color_map,
        title="Session Predictions",
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
    )
    return fig
