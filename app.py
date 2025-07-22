import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="📊 Telco Churn Prediction", layout="centered", page_icon="📉")

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    h1, h2, h3 {color: #003366;}
    .stButton>button {background-color: #003366; color: white; font-weight: bold;}
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-size: 16px;
    }
    .stSelectbox div, .stNumberInput div, .stSlider div {
        font-size: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_artifacts():
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model_data["model"], model_data["feature_names"], encoders

# Prepare inputs
def prepare_input(input_dict, encoders, feature_order):
    df_input = pd.DataFrame([input_dict])
    for col in feature_order:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_order].copy()
    for col, encoder in encoders.items():
        if col in df_input.columns:
            mask_unseen = ~df_input[col].isin(encoder.classes_)
            if mask_unseen.any():
                st.warning(f"⚠️ Unseen category in '{col}': {df_input.loc[mask_unseen, col].values}. Using fallback.")
                df_input.loc[mask_unseen, col] = encoder.classes_[0]
            df_input[col] = encoder.transform(df_input[col])
    return df_input

# Title
st.title("📉 Customer Churn Prediction")
st.markdown("Predict if a customer is likely to churn using Machine Learning. Fill the details below 👇")

# Load model
model, feature_names, encoders = load_artifacts()

# Input section
with st.expander("🡭‍♂️ Customer Details"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", ["0", "1"])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 1)
    with col2:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

with st.expander("📱 Internet & Streaming Services"):
    col3, col4 = st.columns(2)
    with col3:
        OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with col4:
        TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with st.expander("💳 Payment Info"):
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 29.85, 0.05)
    TotalCharges = st.number_input("Total Charges ($)", 0.0, 10000.0, 29.85, 0.05)

# Input dictionary
input_data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}

# Initialize session state
if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = pd.DataFrame(columns=list(input_data.keys()) + ["Prediction", "Churn Probability"])

# Prediction
if st.button("🔍 Predict Churn"):
    input_prepared = prepare_input(input_data, encoders, feature_names)
    pred = model.predict(input_prepared)
    pred_proba = model.predict_proba(input_prepared)

    st.markdown("## 🔎 Prediction Result")
    if pred[0] == 1:
        st.error("❗ **Customer is likely to CHURN**")
    else:
        st.success("✅ **Customer is likely to STAY**")

    st.info(f"🔴 Probability of Churn: **{pred_proba[0, 1]*100:.2f}%**")
    st.info(f"🟢 Probability of Not Churn: **{pred_proba[0, 0]*100:.2f}%**")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pred_proba[0, 1]*100,
        number={'suffix': "%"},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
            'bar': {'color': "red" if pred[0]==1 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "red"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': pred_proba[0,1]*100
            }
        },
        title={'text': "Churn Probability Gauge"},
    ))
    fig.update_layout(height=340, margin=dict(t=50, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

    # Log prediction (safe concat to avoid FutureWarning)
    log_entry = input_data.copy()
    log_entry["Prediction"] = "Churn" if pred[0] == 1 else "No Churn"
    log_entry["Churn Probability"] = round(pred_proba[0, 1]*100, 2)
    new_entry_df = pd.DataFrame([log_entry])

    if st.session_state.prediction_log.empty:
        st.session_state.prediction_log = new_entry_df
    else:
        st.session_state.prediction_log = pd.concat(
            [st.session_state.prediction_log, new_entry_df],
            ignore_index=True
        )

# Dashboard Summary
if not st.session_state.prediction_log.empty:
    st.markdown("## 📊 Dashboard Summary")
    churn_counts = st.session_state.prediction_log["Prediction"].value_counts()
    st.write("Total Predictions:", len(st.session_state.prediction_log))
    st.write("🔴 Churn:", int(churn_counts.get("Churn", 0)))
    st.write("🟢 No Churn:", int(churn_counts.get("No Churn", 0)))

    # Pie Chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=churn_counts.index,
        values=churn_counts.values,
        hole=0.5,
        marker_colors=["#EF553B", "#00CC96"]
    )])
    fig_pie.update_layout(title="Churn vs No Churn Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    # CSV download
    st.markdown("### 📅 Download Prediction History")
    csv = st.session_state.prediction_log.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="churn_prediction_history.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown(
    '<center>🧠 Built with 📊 by <a href="https://www.linkedin.com/in/laxmisingh79" target="_blank">Laxmi Kumari Singh</a> '
    '| <a href="https://github.com/Laxmi-ai" target="_blank">GitHub</a> | Telco Churn Predictor</center>',
    unsafe_allow_html=True
)
