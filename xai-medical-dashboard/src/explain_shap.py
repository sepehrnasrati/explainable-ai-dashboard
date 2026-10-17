from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "baseline_pipeline.joblib"


def load_sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Same dataset as training (for now).
    Later we swap this for Heart Disease / Diabetes dataset.
    """
    from sklearn.datasets import load_breast_cancer

    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run src/train_baseline.py first."
        )

    # Load model pipeline (preprocess + classifier)
    pipeline = joblib.load(MODEL_PATH)

    X, y = load_sample_data()

    # Pick one example patient row
    idx = 0
    x_one = X.iloc[[idx]]

    # Prediction
    pred_class = int(pipeline.predict(x_one)[0])
    pred_proba = float(pipeline.predict_proba(x_one)[0][pred_class])

    print("\n=== Single Prediction ===")
    print(f"Row index: {idx}")
    print(f"Predicted class: {pred_class}")
    print(f"Predicted probability (for predicted class): {pred_proba:.4f}")

    # ---- SHAP Explanation ----
    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X)
    x_one_transformed = preprocessor.transform(x_one)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    explainer = shap.LinearExplainer(
        model, X_transformed, feature_perturbation="interventional"
    )

    shap_values = explainer.shap_values(x_one_transformed)

    if isinstance(shap_values, list):
        shap_vals_for_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_vals_for_plot = shap_values

    shap_vals_for_plot = np.array(shap_vals_for_plot).reshape(1, -1)

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    exp = shap.Explanation(
        values=shap_vals_for_plot[0],
        base_values=base_value,
        data=x_one_transformed.toarray()[0]
        if hasattr(x_one_transformed, "toarray")
        else x_one_transformed[0],
        feature_names=feature_names,
    )

    # ---- SAVE SHAP PLOT AS IMAGE ----
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "shap_waterfall_example.png"

    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"\nSaved SHAP plot to: {out_path}\n")


if __name__ == "__main__":
    main()
