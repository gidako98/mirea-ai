"""Model creation, training and evaluation helpers."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import FEATURE_COLUMNS, TARGET_COLUMN

TEXT_COLUMN = "text"
CATEGORICAL_COLUMNS = ["channel", "customer_tier", "product_area"]
NUMERIC_COLUMNS = ["sentiment_score", "account_age_days"]


def split_features_target(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and target vector from a validated dataframe."""
    return data[FEATURE_COLUMNS].copy(), data[TARGET_COLUMN].copy()


def build_final_model(max_text_features: int = 3000, c_value: float = 2.0, max_iter: int = 1000) -> Pipeline:
    """Build the final scikit-learn pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=max_text_features), TEXT_COLUMN),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )
    classifier = LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])


def build_baseline_model() -> DummyClassifier:
    """Build a simple baseline that predicts the most frequent class."""
    return DummyClassifier(strategy="most_frequent")


def evaluate_model(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Evaluate a trained classifier using accuracy and macro F1."""
    predictions = model.predict(x_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }


def predict_records(model: Any, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Predict priority labels and probabilities for API/CLI records."""
    if not records:
        raise ValueError("At least one record is required")

    frame = pd.DataFrame.from_records(records)
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input fields: {sorted(missing)}")

    frame = frame[FEATURE_COLUMNS].copy()
    labels = model.predict(frame)
    probabilities = model.predict_proba(frame) if hasattr(model, "predict_proba") else None
    classes = list(getattr(model, "classes_", []))

    results: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        item = {"priority": str(label)}
        if probabilities is not None and classes:
            item["probabilities"] = {
                str(cls): round(float(prob), 4) for cls, prob in zip(classes, probabilities[index])
            }
        results.append(item)
    return results
