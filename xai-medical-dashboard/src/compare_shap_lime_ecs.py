from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import joblib

import shap
from lime.lime_tabular import LimeTabularExplainer

from ecs import compute_ecs


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "baseline_pipeline.joblib"


def load_sample_data() -> tuple[pd.DataFrame, pd.Series]:
    from sklearn.datasets import load_breast_cancer
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def clean_shap_feature_name(name: str) -> str:
    """
    Normalize SHAP feature names so they match LIME.
    Examples:
      'num__radius error' -> 'radius error'
      'cat__sex_male'     -> 'sex_male'
    """
    if "__" in name:
        name = name.split("__", 1)[1]
    return name.strip()


def extract_topk_shap_features(
    pipeline,
    X: pd.DataFrame,
    x_one: pd.DataFrame,
    top_k: int,
) -> list[str]:
    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    X_trans = preprocessor.transform(X)
    x_one_trans = preprocessor.transform(x_one)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_trans.shape[1])]

    # CLEAN feature names
    feature_names = [clean_shap_feature_name(n) for n in feature_names]

    explainer = shap.LinearExplainer(
        model,
        X_trans,
        feature_perturbation="interventional",
    )
    shap_values = explainer.shap_values(x_one_trans)

    if isinstance(shap_values, list):
        vals = np.array(
            shap_values[1] if len(shap_values) > 1 else shap_values[0]
        ).reshape(-1)
    else:
        vals = np.array(shap_values).reshape(-1)

    order = np.argsort(np.abs(vals))[::-1]
    return [feature_names[i] for i in order[:top_k]]


def parse_lime_feature_name(lime_rule: str) -> str:
    """
    LIME returns rules like:
      'mean radius > 14.2'
      '12.3 < worst texture <= 18.1'
    Extract just the feature name.
    """
    candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_ ]+", lime_rule)
    if not candidates:
        return lime_rule.strip()
    return max((c.strip() for c in candidates), key=len)


def extract_topk_lime_features(
    pipeline,
    X: pd.DataFrame,
    x_one_series: pd.Series,
    top_k: int,
) -> list[str]:
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
        data_row=x_one_series.values,
        predict_fn=predict_proba_fn,
        num_features=top_k,
    )

    raw_rules = [rule for rule, _ in exp.as_list()]
    return [parse_lime_feature_name(r) for r in raw_rules]


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run src/train_baseline.py first."
        )

    pipeline = joblib.load(MODEL_PATH)
    X, y = load_sample_data()

    idx = 0
    x_one_df = X.iloc[[idx]]
    x_one_series = X.iloc[idx]

    pred_class = int(pipeline.predict(x_one_df)[0])
    pred_proba = float(pipeline.predict_proba(x_one_df)[0][pred_class])

    TOP_KS = [3, 5, 10]
    rows = []

    print("\n=== Prediction ===")
    print(f"Row index: {idx}")
    print(f"Predicted class: {pred_class}")
    print(f"Predicted probability: {pred_proba:.4f}")

    for k in TOP_KS:
        shap_feats = extract_topk_shap_features(
            pipeline, X, x_one_df, top_k=k
        )
        lime_feats = extract_topk_lime_features(
            pipeline, X, x_one_series, top_k=k
        )

        result = compute_ecs(shap_feats, lime_feats, top_k=k)

        print(f"\n=== ECS @ top-{k} ===")
        print(f"SHAP: {result.shap_features}")
        print(f"LIME: {result.lime_features}")
        print(f"Intersection: {result.intersection}")
        print(f"ECS: {result.ecs:.3f}")

        rows.append({
            "idx": idx,
            "top_k": k,
            "pred_class": pred_class,
            "pred_proba": pred_proba,
            "shap_topk": "|".join(result.shap_features),
            "lime_topk": "|".join(result.lime_features),
            "intersection": "|".join(result.intersection),
            "ecs": result.ecs,
        })

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "ecs_single_example.csv"

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n✅ Saved ECS results to: {out_path}\n")


if __name__ == "__main__":
    main()
