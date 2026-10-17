from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from lime.lime_tabular import LimeTabularExplainer


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "baseline_pipeline.joblib"


def load_sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Same dataset as training (for now).
    Later we swap to UCI Heart Disease / Diabetes.
    """
    from sklearn.datasets import load_breast_cancer

    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run src/train_baseline.py first.")

    pipeline = joblib.load(MODEL_PATH)

    X, y = load_sample_data()

    # Use same patient row as SHAP script
    idx = 0
    x_one = X.iloc[idx]         # Series
    x_one_df = X.iloc[[idx]]    # 1-row DataFrame (for prediction)

    pred_class = int(pipeline.predict(x_one_df)[0])
    pred_proba = float(pipeline.predict_proba(x_one_df)[0][pred_class])

    print("\n=== Single Prediction (LIME) ===")
    print(f"Row index: {idx}")
    print(f"Predicted class: {pred_class}")
    print(f"Predicted probability (for predicted class): {pred_proba:.4f}")

    # LIME needs a prediction function that accepts a 2D numpy array
    # We'll wrap the pipeline.predict_proba and convert array->DataFrame with correct columns.
    feature_names = X.columns.tolist()

    def predict_proba_fn(x_arr: np.ndarray) -> np.ndarray:
        x_df = pd.DataFrame(x_arr, columns=feature_names)
        return pipeline.predict_proba(x_df)

    explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=feature_names,
        class_names=["class_0", "class_1"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    exp = explainer.explain_instance(
        data_row=x_one.values,
        predict_fn=predict_proba_fn,
        num_features=10
    )

    # Print top features in terminal
    print("\nTop LIME features (weight):")
    for feat, weight in exp.as_list():
        print(f"{feat:50s} {weight:+.4f}")

    # Save explanation as HTML (best for report/demo)
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "lime_explanation_example.html"
    exp.save_to_file(str(out_path))
    print(f"\n✅ Saved LIME explanation to: {out_path}\n")


if __name__ == "__main__":
    main()
