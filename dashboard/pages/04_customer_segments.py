"""Customer segments page."""

import streamlit as st

from i18n import t, render_language_selector

st.set_page_config(page_title=t("page_title_customer_segments"), page_icon="👥")

with st.sidebar:
    render_language_selector()

st.title(f"👥 {t('segments_title')}")

st.markdown(
    """
    ### {title}
    
    {desc}
    """.format(
        title=t("segments_title"),
        desc=t("segments_desc"),
    )
)

st.subheader(t("segment_overview"))
st.info(t("segment_overview_placeholder"))

st.subheader(t("segment_churn_rates"))
st.info(t("segment_churn_rates_placeholder"))

st.subheader(t("segment_characteristics"))
st.info(t("segment_characteristics_placeholder"))
