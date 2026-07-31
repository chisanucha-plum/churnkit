"""Model performance page."""

import streamlit as st
from i18n import render_language_selector, t

st.set_page_config(page_title=t("page_title_model_performance"), page_icon="📊")

with st.sidebar:
    render_language_selector()

st.title(f"📊 {t('model_title')}")

st.markdown(
    """
    ### {title}
    
    {desc}
    """.format(
        title=t("model_title"),
        desc=t("model_desc"),
    )
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(t("accuracy"), "92.3%", "+1.2%")

with col2:
    st.metric(t("precision"), "89.5%", "+0.8%")

with col3:
    st.metric(t("recall"), "85.2%", "+2.1%")

with col4:
    st.metric(t("f1_score"), "87.3%", "+1.5%")

st.divider()

st.subheader(t("confusion_matrix"))
st.info(t("confusion_matrix_placeholder"))

st.subheader(t("roc_curve"))
st.info(t("roc_curve_placeholder"))

st.subheader(t("model_versions"))
st.info(t("model_versions_placeholder"))
