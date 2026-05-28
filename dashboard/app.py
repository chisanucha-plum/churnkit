"""Streamlit dashboard main app."""

import streamlit as st

from i18n import t, render_language_selector

st.set_page_config(
    page_title=t("page_title_dashboard"),
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    render_language_selector()

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .stApp {
            background: radial-gradient(1200px circle at 10% 10%, #121826 0%, #0b0f14 40%, #070a0e 100%);
            color: #e5e7eb;
        }
        .hero {
            background: linear-gradient(135deg, #151b26 0%, #0f1720 100%);
            border: 1px solid #1f2937;
            border-radius: 18px;
            padding: 24px 28px;
            margin-bottom: 24px;
            box-shadow: 0 12px 28px rgba(3, 7, 18, 0.6);
        }
        .hero h1 { font-family: "Georgia", "Times New Roman", serif; color: #f9fafb; }
        .hero p { font-size: 1.05rem; color: #cbd5f5; }
        .callout {
            border-left: 4px solid #38bdf8;
            background: rgba(56, 189, 248, 0.08);
            padding: 12px 14px;
            border-radius: 10px;
            color: #e2e8f0;
        }
        .section-title {
            font-family: "Georgia", "Times New Roman", serif;
            letter-spacing: 0.2px;
        }
        .stMarkdown, .stMarkdown p, .stMarkdown li { color: #e5e7eb; }
        .stInfo { background: rgba(15, 23, 42, 0.6); border: 1px solid #1f2937; }
        .stDivider { border-color: #1f2937; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
        <div class="callout">{tip}</div>
    </div>
    """.format(
        title=t("dashboard_title"),
        subtitle=t("dashboard_subtitle"),
        tip=t("dashboard_tip"),
    ),
    unsafe_allow_html=True,
)

st.markdown(f"### {t('quick_navigation')}")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        "- {form}\n- {exec}\n- {risk}".format(
            form=t("nav_prediction_form"),
            exec=t("nav_executive_summary"),
            risk=t("nav_high_risk_customers"),
        )
    )
with col2:
    st.markdown(
        "- {feature}\n- {segments}\n- {performance}".format(
            feature=t("nav_feature_analysis"),
            segments=t("nav_customer_segments"),
            performance=t("nav_model_performance"),
        )
    )
with col3:
    st.markdown(
        "- {ai}\n- {metrics}\n- {data}".format(
            ai=t("nav_ai_insights"),
            metrics=t("nav_metrics_health"),
            data=t("nav_data_overview"),
        )
    )

st.divider()

st.markdown(f"### {t('what_you_can_do')}")
st.markdown(
    """
    - {line1}
    - {line2}
    - {line3}
    """.format(
        line1=t("what_you_can_do_1"),
        line2=t("what_you_can_do_2"),
        line3=t("what_you_can_do_3"),
    )
)

st.info(t("sidebar_hint"))
