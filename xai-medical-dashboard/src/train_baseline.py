from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib


@dataclass
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = root / "data"
    models_dir: Path = root / "models"


def main() -> None:
    """
    Stage 1: Baseline ML pipeline using the HEART dataset.
    """

    # ---- UPDATED SECTION ----
    from data_loader import load_heart_data

    paths = Paths()
    paths.models_dir.mkdir(exist_ok=True)

    # Loader now returns full dataframe plus X and y
    df, X, y = load_heart_data()
    # ------------------------

    # Identify numeric vs categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=2000)

    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n✅ Baseline heart-disease model trained.")
    print(f"Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, preds))

    # Save pipeline (preprocess + model together)
    out_path = paths.models_dir / "baseline_pipeline.joblib"
    joblib.dump(clf, out_path)
    print(f"\nSaved model to: {out_path}\n")


if __name__ == "__main__":
    main()
