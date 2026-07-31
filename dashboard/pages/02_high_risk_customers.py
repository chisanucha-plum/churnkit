"""High risk customers page."""

import streamlit as st
from i18n import render_language_selector, t

st.set_page_config(page_title=t("page_title_high_risk"), page_icon="⚠️")

with st.sidebar:
    render_language_selector()

st.title(f"⚠️ {t('high_risk_title')}")

st.markdown(
    """
    ### {title}
    
    {desc}
    """.format(
        title=t("high_risk_title"),
        desc=t("high_risk_desc"),
    )
)

st.subheader(t("high_risk_list"))
st.info(t("high_risk_list_placeholder"))

st.subheader(t("risk_factors"))
st.info(t("risk_factors_placeholder"))

st.subheader(t("recommended_actions"))
st.info(t("recommended_actions_placeholder"))
