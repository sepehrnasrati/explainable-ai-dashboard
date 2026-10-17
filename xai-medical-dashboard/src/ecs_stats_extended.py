import pandas as pd
from pathlib import Path
import joblib

from data_loader import load_heart_data
from compare_shap_lime_ecs import extract_topk_shap_features, extract_topk_lime_features
from ecs import compute_ecs

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "baseline_pipeline.joblib"

pipeline = joblib.load(MODEL_PATH)
df, X, y = load_heart_data()

TOP_K = 5  # use the same K as your dashboard

ecs_scores = []

for i in range(len(X)):
    x_df = X.iloc[[i]]
    x_series = X.iloc[i]

    shap_feats = extract_topk_shap_features(pipeline, X, x_df, TOP_K)
    lime_feats = extract_topk_lime_features(pipeline, X, x_series, TOP_K)

    result = compute_ecs(shap_feats, lime_feats, TOP_K)
    ecs_scores.append(result.ecs)

ecs_series = pd.Series(ecs_scores)

total = len(ecs_series)
below_05 = (ecs_series < 0.5).sum()
perfect = (ecs_series == 1.0).sum()

print("Total cases:", total)
print("Mean ECS:", round(ecs_series.mean(), 4))
print("Min ECS:", ecs_series.min())
print("Max ECS:", ecs_series.max())
print("% with ECS < 0.5:", round((below_05 / total) * 100, 2))
print("% with ECS = 1.0:", round((perfect / total) * 100, 2))
