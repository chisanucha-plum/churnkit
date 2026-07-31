"""Chart components."""

import plotly.graph_objects as go
import streamlit as st


def display_churn_distribution(data: dict):
    """Display churn distribution chart."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Churned", "Retained"],
                y=[data.get("churned", 0), data.get("retained", 0)],
                marker_color=["#FF6B6B", "#4ECDC4"],
            )
        ]
    )

    fig.update_layout(
        title="Customer Churn Distribution",
        xaxis_title="Status",
        yaxis_title="Count",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def display_risk_distribution(data: dict):
    """Display risk level distribution."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["High Risk", "Medium Risk", "Low Risk"],
                values=[
                    data.get("high_risk", 0),
                    data.get("medium_risk", 0),
                    data.get("low_risk", 0),
                ],
                marker_colors=["#FF6B6B", "#FFA500", "#4ECDC4"],
            )
        ]
    )

    fig.update_layout(title="Risk Level Distribution", height=400)

    st.plotly_chart(fig, use_container_width=True)
