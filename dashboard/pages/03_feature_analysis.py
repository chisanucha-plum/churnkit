"""Feature analysis page."""

import streamlit as st

from i18n import t, render_language_selector

st.set_page_config(page_title=t("page_title_feature_analysis"), page_icon="🔍")

with st.sidebar:
    render_language_selector()

st.title(f"🔍 {t('feature_title')}")

st.markdown(
    """
    ### {title}
    
    {desc}
    """.format(
        title=t("feature_title"),
        desc=t("feature_desc"),
    )
)

st.subheader(t("feature_importance"))
st.info(t("feature_importance_placeholder"))

st.subheader(t("feature_correlations"))
st.info(t("feature_correlations_placeholder"))

st.subheader(t("feature_distributions"))
st.info(t("feature_distributions_placeholder"))
