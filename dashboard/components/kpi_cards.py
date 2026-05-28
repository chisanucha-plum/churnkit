"""KPI card components."""

import streamlit as st


def display_kpi_card(title: str, value: str, delta: str = None, color: str = "blue"):
    """Display KPI card."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.metric(label=title, value=value, delta=delta)

    return col1


def display_kpi_row(kpis: list):
    """Display row of KPI cards."""
    cols = st.columns(len(kpis))

    for col, kpi in zip(cols, kpis):
        with col:
            st.metric(label=kpi["title"], value=kpi["value"], delta=kpi.get("delta"))
