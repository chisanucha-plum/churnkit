"""Executive summary page."""

import streamlit as st
from i18n import render_language_selector, t

st.set_page_config(page_title=t("page_title_executive_summary"), page_icon="📈")

with st.sidebar:
    render_language_selector()

st.title(f"📈 {t('exec_title')}")

st.markdown(
    """
    ### {title}
    
    {desc}
    """.format(
        title=t("exec_key_metrics"),
        desc=t("exec_desc"),
    )
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(t("total_customers"), "10,000", "+5%")

with col2:
    st.metric(t("churn_rate"), "26.5%", "-2%")

with col3:
    st.metric(t("high_risk"), "2,650", "+10%")

with col4:
    st.metric(t("avg_confidence"), "92.3%", "+1%")

st.divider()

st.subheader(t("churn_trends"))
st.info(t("churn_trends_placeholder"))
