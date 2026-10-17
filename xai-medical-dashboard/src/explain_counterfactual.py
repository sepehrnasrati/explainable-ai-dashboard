from __future__ import annotations

from pathlib import Path
import pandas as pd
import joblib
import dice_ml
from dice_ml import Dice

from data_loader import load_heart_data


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "baseline_pipeline.joblib"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python src/train_baseline.py")

    # Load heart dataset (same data used for training)
    df, X, y = load_heart_data()

    # Load trained pipeline
    pipeline = joblib.load(MODEL_PATH)

    # Tell DiCE which features are continuous
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

    # Pick a query instance
    idx = 0
    query_instance = df.drop(columns=["target"]).iloc[[idx]]

    print("\n=== Query patient ===")
    print(query_instance)

    cf = exp.generate_counterfactuals(
        query_instance,
        total_CFs=3,
        desired_class="opposite",
        features_to_vary="all",
    )

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "counterfactual_example.csv"

    cf_df = cf.cf_examples_list[0].final_cfs_df
    cf_df.to_csv(out_path, index=False)

    print("\n=== Counterfactuals ===")
    print(cf_df)
    print(f"\n✅ Saved counterfactuals to: {out_path}")


if __name__ == "__main__":
    main()
