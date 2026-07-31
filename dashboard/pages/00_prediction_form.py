"""Prediction form page."""

import json

import requests
import streamlit as st
from i18n import render_language_selector, t

st.set_page_config(page_title=t("page_title_prediction_form"), page_icon="🧾")

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        .panel {
            background: #ffffff;
            border: 1px solid #e5eaf0;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            color: #1f2a37;
        }
        .panel h2 {
            font-family: "Georgia", "Times New Roman", serif;
            color: #111827;
        }
        .panel p {
            color: #334155;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 999px;
            background: #eef3f7;
            color: #1f2a37;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="panel">
        <h2>{title}</h2>
        <p>{desc}</p>
        <span class="badge">{badge}</span>
    </div>
    """.format(
        title=t("prediction_title"),
        desc=t("prediction_desc"),
        badge=t("api_required"),
    ),
    unsafe_allow_html=True,
)

with st.sidebar:
    render_language_selector()
    st.subheader(t("api_settings"))
    api_url = st.text_input(t("prediction_api_url"), value="http://localhost:8000/predict")
    timeout_s = st.slider(t("timeout_seconds"), min_value=2, max_value=30, value=8)

st.divider()

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(t("customer_profile"))

    tenure = st.number_input(t("tenure_months"), min_value=0, max_value=120, value=12)
    monthly_charges = st.number_input(
        t("monthly_charges"), min_value=0.0, max_value=500.0, value=89.5, step=0.1
    )
    total_charges = st.number_input(
        t("total_charges"), min_value=0.0, max_value=10000.0, value=2148.0, step=1.0
    )

    contract_type_options = ["Month-to-month", "One year", "Two year"]
    contract_type_labels = {
        "Month-to-month": t("contract_month_to_month"),
        "One year": t("contract_one_year"),
        "Two year": t("contract_two_year"),
    }
    contract_type = st.selectbox(
        t("contract_type"),
        options=contract_type_options,
        index=0,
        format_func=lambda v: contract_type_labels.get(v, v),
    )

    internet_options = ["Fiber optic", "DSL", "No"]
    internet_labels = {
        "Fiber optic": t("internet_fiber"),
        "DSL": t("internet_dsl"),
        "No": t("internet_no"),
    }
    internet_service = st.selectbox(
        t("internet_service"),
        options=internet_options,
        index=0,
        format_func=lambda v: internet_labels.get(v, v),
    )

    payment_options = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    payment_labels = {
        "Electronic check": t("payment_electronic"),
        "Mailed check": t("payment_mailed"),
        "Bank transfer": t("payment_bank"),
        "Credit card": t("payment_credit"),
    }
    payment_method = st.selectbox(
        t("payment_method"),
        options=payment_options,
        index=0,
        format_func=lambda v: payment_labels.get(v, v),
    )

with col_right:
    st.subheader(t("services_flags"))

    def yes_no(label_key: str, default: int = 0) -> int:
        options = [0, 1]
        labels = {0: t("no"), 1: t("yes")}
        return st.selectbox(
            t(label_key),
            options,
            index=default,
            format_func=lambda v: labels.get(v, v),
        )

    online_security = yes_no("online_security", 0)
    online_backup = yes_no("online_backup", 0)
    device_protection = yes_no("device_protection", 0)
    tech_support = yes_no("tech_support", 0)
    streaming_tv = yes_no("streaming_tv", 1)
    streaming_movies = yes_no("streaming_movies", 1)
    paperless_billing = yes_no("paperless_billing", 1)
    senior_citizen = yes_no("senior_citizen", 0)
    partner = yes_no("partner", 1)
    dependents = yes_no("dependents", 0)
    phone_service = yes_no("phone_service", 1)
    multiple_lines = yes_no("multiple_lines", 0)

payload = {
    "tenure": int(tenure),
    "monthly_charges": float(monthly_charges),
    "total_charges": float(total_charges),
    "contract_type": contract_type,
    "internet_service": internet_service,
    "online_security": online_security,
    "online_backup": online_backup,
    "device_protection": device_protection,
    "tech_support": tech_support,
    "streaming_tv": streaming_tv,
    "streaming_movies": streaming_movies,
    "payment_method": payment_method,
    "paperless_billing": paperless_billing,
    "senior_citizen": senior_citizen,
    "partner": partner,
    "dependents": dependents,
    "phone_service": phone_service,
    "multiple_lines": multiple_lines,
}

st.divider()

col_a, col_b = st.columns([1, 2])

with col_a:
    submitted = st.button(t("run_prediction"), type="primary", use_container_width=True)

with col_b:
    with st.expander(t("preview_payload")):
        st.code(json.dumps(payload, indent=2), language="json")

if submitted:
    try:
        response = requests.post(api_url, json=payload, timeout=timeout_s)
        response.raise_for_status()
        result = response.json()

        st.success(t("prediction_completed"))

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(t("churn_probability"), f"{result.get('churn_probability', 0):.3f}")
        with m2:
            st.metric(t("risk_level"), str(result.get("risk_level", "N/A")))
        with m3:
            st.metric(t("risk_score"), str(result.get("risk_score", "N/A")))

        st.subheader(t("recommendation"))
        st.info(result.get("recommendation", t("no_recommendation")))

    except requests.exceptions.RequestException as exc:
        st.error(t("prediction_failed"))
        st.write(str(exc))
