import sys
from pathlib import Path

# ensure src folder is on Python path 
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))


import streamlit as st
import pandas as pd
import joblib

# imports for visual explanations
import shap
import matplotlib.pyplot as plt
from streamlit.components.v1 import html

import dice_ml
from dice_ml import Dice

from data_loader import load_heart_data
from compare_shap_lime_ecs import (
    extract_topk_shap_features,
    extract_topk_lime_features,
)
from ecs import compute_ecs

MODEL_PATH = ROOT / "models" / "baseline_pipeline.joblib"


# shap helper function
def get_shap_plot(pipeline, X, x_one_df):
    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    X_trans = preprocessor.transform(X)
    x_one_trans = preprocessor.transform(x_one_df)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()

        # remove num__ and cat__ prefixes
        feature_names = [f.split("__")[-1] for f in feature_names]

    except:
        feature_names = X.columns.tolist()

    explainer = shap.LinearExplainer(
        model,
        X_trans,
        feature_perturbation="interventional"
    )

    shap_values = explainer.shap_values(x_one_trans)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    shap_vals = shap_vals.reshape(1, -1)

    exp = shap.Explanation(
        values=shap_vals[0],
        base_values=explainer.expected_value,
        data=x_one_trans[0],
        feature_names=feature_names,
    )

    fig = plt.figure()
    shap.plots.waterfall(exp, max_display=10, show=False)
    return fig



# lime helper function
def get_lime_html(pipeline, X, x_one_series):
    from lime.lime_tabular import LimeTabularExplainer

    feature_names = X.columns.tolist()

    def predict_proba_fn(x_arr):
        x_df = pd.DataFrame(x_arr, columns=feature_names)
        return pipeline.predict_proba(x_df)

    explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=feature_names,
        mode="classification"
    )

    exp = explainer.explain_instance(
        x_one_series.values,
        predict_proba_fn,
        num_features=10
    )

    return exp.as_html()



# LIVE counterfactual generation
def generate_counterfactuals_live(pipeline, df, idx):
    continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    dice_data = dice_ml.Data(
        dataframe=df,
        continuous_features=continuous_features,
        outcome_name="target",
    )

    dice_model = dice_ml.Model(
        model=pipeline,
        backend="sklearn",
        model_type="classifier",
    )

    exp = Dice(dice_data, dice_model, method="random")

    query_instance = df.drop(columns=["target"]).iloc[[idx]]

    cf = exp.generate_counterfactuals(
        query_instance,
        total_CFs=3,
        desired_class="opposite",
    )

    return cf.cf_examples_list[0].final_cfs_df



# page configuration
st.set_page_config(page_title="XAI Medical Dashboard", layout="wide")
st.title("Comparative Explainable AI Dashboard")

# dashboard explaination
st.info(
    "This dashboard compares two explainability methods (SHAP and LIME) "
    "for medical predictions and measures their agreement using the "
    "Explanation Consensus Score (ECS)."
)



# load model and data
pipeline = joblib.load(MODEL_PATH)
df, X, y = load_heart_data()

st.sidebar.header("Select Patient")

idx = st.sidebar.selectbox(
    "Patient Index",
    options=list(range(len(df)))
)

x_one_df = X.iloc[[idx]]
x_one_series = X.iloc[idx]

pred_class = int(pipeline.predict(x_one_df)[0])
pred_proba = float(pipeline.predict_proba(x_one_df)[0][pred_class])

st.subheader("Prediction")
st.write(f"**Predicted Class:** {pred_class}")
st.write(f"**Probability:** {pred_proba:.3f}")

TOP_K = st.sidebar.slider("Top-K Features for ECS", 3, 10, 5)

shap_feats = extract_topk_shap_features(pipeline, X, x_one_df, TOP_K)
lime_feats = extract_topk_lime_features(pipeline, X, x_one_series, TOP_K)

result = compute_ecs(shap_feats, lime_feats, TOP_K)

col1, col2 = st.columns(2)

with col1:
    st.subheader("SHAP Top Features")
    st.write(shap_feats)

with col2:
    st.subheader("LIME Top Features")
    st.write(lime_feats)

st.subheader("Explanation Consensus Score (ECS)")

color = "green" if result.ecs > 0.6 else "orange" if result.ecs > 0.3 else "red"

st.markdown(
    f"<h3 style='color:{color}'>ECS = {result.ecs:.2f}</h3>",
    unsafe_allow_html=True
)

st.write("Intersection:", result.intersection)


# SHAP visualisation
st.subheader("SHAP Explanation Plot")

fig = get_shap_plot(pipeline, X, x_one_df)
st.pyplot(fig)



# LIME visual
st.subheader("LIME Explanation")

lime_html = get_lime_html(pipeline, X, x_one_series)

styled_html = f"""
<div style="
    background-color: white;
    color: black;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ccc;
">
{lime_html}
</div>
"""

html(styled_html, height=600, scrolling=True)



# LIVE counterfactuals
st.subheader("Counterfactual Explanations")

if st.button("Generate Counterfactuals"):
    with st.spinner("Generating counterfactuals..."):
        cf_df = generate_counterfactuals_live(pipeline, df, idx)
        st.dataframe(cf_df)
