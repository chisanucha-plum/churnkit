"""AI insights page."""

import streamlit as st
from i18n import render_language_selector, t

st.set_page_config(page_title=t("page_title_ai_insights"), page_icon="✨")

with st.sidebar:
    render_language_selector()

st.title(f"✨ {t('ai_title')}")

st.markdown(
    """
    ### {title}
    
    {desc}
    """.format(
        title=t("ai_title"),
        desc=t("ai_desc"),
    )
)

st.subheader(t("retention_strategies"))
st.info(t("retention_strategies_placeholder"))

st.subheader(t("customer_insights"))
st.info(t("customer_insights_placeholder"))

st.subheader(t("predictive_recommendations"))
st.info(t("predictive_recommendations_placeholder"))
