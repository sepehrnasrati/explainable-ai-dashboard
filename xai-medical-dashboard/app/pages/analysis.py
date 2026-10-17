import sys
from pathlib import Path

# --- FIX PATHS (analysis.py is inside app/pages) ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
# ---------------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_heart_data
from compare_shap_lime_ecs import (
    extract_topk_shap_features,
    extract_topk_lime_features,
)
from ecs import compute_ecs

import joblib

MODEL_PATH = ROOT / "models" / "baseline_pipeline.joblib"

st.set_page_config(page_title="ECS Analysis", layout="wide")

st.title("Dataset-Level ECS Analysis")

# Load model and dataset
pipeline = joblib.load(MODEL_PATH)
df, X, y = load_heart_data()

TOP_K = st.sidebar.slider("Top-K Features", 3, 10, 5)

st.write("Computing ECS for all patients...")

ecs_scores = []

progress = st.progress(0)

for i in range(len(X)):
    x_df = X.iloc[[i]]
    x_series = X.iloc[i]

    shap_feats = extract_topk_shap_features(pipeline, X, x_df, TOP_K)
    lime_feats = extract_topk_lime_features(pipeline, X, x_series, TOP_K)

    result = compute_ecs(shap_feats, lime_feats, TOP_K)
    ecs_scores.append(result.ecs)

    progress.progress((i + 1) / len(X))

df_scores = pd.DataFrame({"ECS": ecs_scores})

st.subheader("ECS Distribution")

fig, ax = plt.subplots()
ax.hist(df_scores["ECS"], bins=10, color="skyblue")
ax.set_xlabel("ECS Score")
ax.set_ylabel("Number of Patients")
ax.set_title("Distribution of ECS Across Dataset")

st.pyplot(fig)

st.subheader("Statistics")

st.write(df_scores.describe())

st.subheader("Patients with Lowest Agreement")

low = df_scores.sort_values("ECS").head(10)
st.dataframe(low)

# ---------------------------------------------------
# STEP 2 – EXPORT FUNCTIONALITY
# ---------------------------------------------------

st.subheader("Export Results")

csv = df_scores.to_csv(index=False)

st.download_button(
    label="Download ECS Results as CSV",
    data=csv,
    file_name="ecs_results.csv",
    mime="text/csv"
)
