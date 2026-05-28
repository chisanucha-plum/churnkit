"""Table components."""

import streamlit as st
import pandas as pd


def display_customers_table(customers: list):
    """Display customers table."""
    if not customers:
        st.info("No customers to display")
        return

    df = pd.DataFrame(customers)
    st.dataframe(df, use_container_width=True)


def display_predictions_table(predictions: list):
    """Display predictions table."""
    if not predictions:
        st.info("No predictions to display")
        return

    df = pd.DataFrame(predictions)
    st.dataframe(df, use_container_width=True)
