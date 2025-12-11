import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import joblib # type: ignore
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.preprocess import preprocess_test # type: ignore
from plots.evaluate import evaluate_model # type: ignore 


st.set_page_config(page_title="Anomaly Detector", layout="wide")

# =========================
# Centered Title & Description
# =========================
st.markdown(
    "<div style='text-align: center'>"
    "<h1>User Behavior Anomaly Detection Dashboard</h1>"
    "<p>This dashboard demonstrates a simple anomaly detection model using <b>Isolation Forest</b></p>"
    "<p>Use the button below to run the demo using the built-in sample dataset.</p>"
    "</div>",
    unsafe_allow_html=True
)

# ================================================
# LOAD MODEL
# ================================================
@st.cache_resource
def load_model():
    return joblib.load("models/isolation_forest.pkl")

model = load_model()

# ================================================
# LOAD TEST DATA
# ================================================
@st.cache_data
def load_sample_data():
    return pd.read_csv("data/test_preprocessed.csv")

# Centered Load Sample Dataset Button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("📥 Load Sample Dataset"):
        df = load_sample_data()
        st.success("Dataset loaded successfully!", icon="✅")
        st.dataframe(df.head(), width=825)

# ================================================
# RUN ANALYSIS
# ================================================
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("⚙️ Run Preprocessing + Scoring"):
        df = load_sample_data()

        raw_columns = {"timestamp", "device_type", "ip"}
        if raw_columns.issubset(df.columns):
            processed, labels = preprocess_test(df)
        else:
            processed = df.copy()
            for col in ["label", "user_id", "location"]:
                if col in processed.columns:
                    processed = processed.drop(columns=[col])

        anomaly_scores = model.decision_function(processed)
        preds = model.predict(processed)

        processed["anomaly_score"] = -anomaly_scores
        processed["prediction"] = preds
        processed["anomaly_label"] = processed["prediction"].map({-1: "Anomaly", 1: "Normal"})

        st.session_state["processed"] = processed
        st.success("Analysis complete!", icon="✅")
        st.dataframe(processed.head(), width=825)

# ================================================
# VISUALIZATION SECTION
# ================================================
st.markdown("<h2 style='text-align: center'>📊 Visualizations</h2>", unsafe_allow_html=True)

if "processed" in st.session_state:
    df = st.session_state["processed"]

    # 1. Anomaly Score Distribution
    st.markdown("<h3 style='text-align: center'>Anomaly Score Distribution</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df["anomaly_score"], bins=50, ax=ax)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=True)

    # 2. Score Scatter Plot
    st.markdown("<h3 style='text-align: center'>Scatter Plot (Score vs Session Duration)</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=df, x="session_duration", y="anomaly_score", hue="anomaly_label", ax=ax)
    ax.set_xlabel("Session Duration")
    ax.set_ylabel("Anomaly Score")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=True)

    # 3. Top 20 Most Risky Users
    st.markdown("<h3 style='text-align: center'>Top 20 Most Risky Users</h3>", unsafe_allow_html=True)
    top = df.sort_values("anomaly_score", ascending=False).head(20)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.dataframe(top, width=825)

# ================================================
# EVALUATION SECTION
# ================================================
st.markdown("<h2 style='text-align: center'>📈 Model Evaluation</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("Run Evaluation on Test Split"):
        metrics = evaluate_model()
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Detection Rate", f"{metrics['detection_rate']:.2f}")
        with metric_col2:
            st.metric("False Positive Rate", f"{metrics['false_positive_rate']:.2f}")
        st.success("Evaluation complete!", icon="✅")

# ================================================
# RAW DATA TABLE
# ================================================
st.markdown("<h2 style='text-align: center'>📄 Raw Scored Data</h2>", unsafe_allow_html=True)
if "processed" in st.session_state:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.dataframe(st.session_state["processed"], width=825)

# ================================================
# EXPORT
# ================================================
if "processed" in st.session_state:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("⬇️ Download Results as CSV"):
            csv = st.session_state["processed"].to_csv(index=False)
            st.download_button("Download", csv, "results.csv", "text/csv")
