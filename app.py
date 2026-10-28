
"""Application entry point for customer churn prediction."""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Customer Churn Insights",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.joblib")


saved = load_model()

model = saved["model"]
threshold = saved["threshold"]
features = saved["features"]


# --------------------------------------------------
# Application header
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict customer churn risk and explore "
    "possible customer-retention actions."
)

st.info(
    f"Model: Logistic Regression | "
    f"Decision threshold: {threshold:.2f}"
)


# --------------------------------------------------
# Customer input form
# --------------------------------------------------

st.header("Enter Customer Details")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Has Partner?", ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Has Dependents?", ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (months)", 0, 72, 12
        )

        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=100000.0,
            value=monthly_charges * tenure,
            step=10.0
        )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# Prediction results
# --------------------------------------------------

if submitted:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([customer])
    input_df = input_df[features]

    probability = model.predict_proba(input_df)[0][1]
    prediction = probability >= threshold

    st.divider()
    st.header("Prediction Results")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Estimated Churn Probability",
            f"{probability:.1%}"
        )

    with c2:
        st.metric(
            "Decision Threshold",
            f"{threshold:.0%}"
        )

    if prediction:
        st.error("⚠️ Churn Risk: HIGH")

        st.subheader("Suggested Retention Actions")

        suggestions = []

        if contract == "Month-to-month":
            suggestions.append(
                "Consider offering a suitable longer-term contract."
            )

        if tenure <= 12:
            suggestions.append(
                "Provide onboarding support and check customer satisfaction."
            )

        if monthly_charges >= 80:
            suggestions.append(
                "Review whether a more suitable plan is available."
            )

        if payment_method == "Electronic check":
            suggestions.append(
                "Offer help exploring convenient automatic payment options."
            )

        if suggestions:
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
        else:
            st.write(
                "Review the customer's service experience and "
                "contact them to understand any concerns."
            )

        st.caption(
            "These are general suggestions, not guarantees of retention."
        )

    else:
        st.success("✅ Churn Risk: LOW")

        st.write(
            "Continue regular customer engagement and monitor "
            "changes in customer behaviour."
        )

    st.caption(
        "This prediction is based on a public historical dataset. "
        "It is an estimate, not a guarantee of future behaviour."
    )


# --------------------------------------------------
# Model explainability
# --------------------------------------------------

st.divider()
st.header("🔍 Understanding the Model")

st.write(
    "Explore how the trained Logistic Regression model "
    "associates input features with customer churn."
)

try:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })

    feature_df["AbsoluteCoefficient"] = (
        feature_df["Coefficient"].abs()
    )

    feature_df = feature_df.sort_values(
        "AbsoluteCoefficient",
        ascending=False
    )

    top_features = (
        feature_df.head(15)
        .sort_values("Coefficient")
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        top_features["Feature"],
        top_features["Coefficient"]
    )

    ax.axvline(
        0,
        color="black",
        linewidth=0.8
    )

    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_ylabel("Feature")
    ax.set_title("Top 15 Features by Absolute Coefficient")

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Positive coefficients increase predicted churn odds, "
        "while negative coefficients decrease them, holding "
        "other encoded features constant. These are associations, "
        "not proof of causation."
    )

    with st.expander("View coefficient details"):

        display_df = feature_df[
            ["Feature", "Coefficient"]
        ].head(15).copy()

        display_df["Coefficient"] = (
            display_df["Coefficient"].round(4)
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

except (KeyError, AttributeError, ValueError) as error:
    st.warning(
        "Could not display model coefficients. "
        "Please check the saved model's pipeline step names."
    )

    st.caption(f"Technical details: {error}")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction & Retention Insights System | "
    "Machine Learning Portfolio Project"
)
