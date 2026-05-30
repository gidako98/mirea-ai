"""Train baseline and final models for ticket priority prediction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data import TARGET_COLUMN, generate_synthetic_tickets, load_dataset, save_dataset
from src.modeling import build_baseline_model, build_final_model, evaluate_model, split_features_target

LOGGER = logging.getLogger(__name__)


def train_from_config(config_path: str | Path = "configs/config.json") -> dict:
    """Train the model, save artifacts and return metrics."""
    config = load_config(config_path)
    data_path = Path(config["data_path"])

    if not data_path.exists():
        LOGGER.info("Dataset not found at %s; generating synthetic data", data_path)
        data = generate_synthetic_tickets(rows=600, seed=int(config["random_state"]))
        save_dataset(data, data_path)
    else:
        data = load_dataset(data_path)

    x, y = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(config["test_size"]),
        random_state=int(config["random_state"]),
        stratify=y,
    )

    baseline = build_baseline_model()
    baseline.fit(x_train, y_train)
    baseline_metrics = evaluate_model(baseline, x_test, y_test)

    model_cfg = config["model"]
    final_model = build_final_model(
        max_text_features=int(model_cfg["max_text_features"]),
        c_value=float(model_cfg["logistic_regression_C"]),
        max_iter=int(model_cfg["max_iter"]),
    )
    final_model.fit(x_train, y_train)
    final_metrics = evaluate_model(final_model, x_test, y_test)

    model_path = Path(config["model_path"])
    metrics_path = Path(config["metrics_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)

    metrics = {
        "dataset_rows": int(len(data)),
        "target_distribution": data[TARGET_COLUMN].value_counts().to_dict(),
        "baseline": baseline_metrics,
        "final_model": final_metrics,
        "selected_model": "tfidf_onehot_numeric_logistic_regression",
        "selection_reason": "Final model has higher macro F1 than the majority-class baseline while remaining interpretable and fast.",
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    model_card_path = model_path.parent / "model_card.md"
    model_card_path.write_text(
        "# Model card\n\n"
        "## Purpose\n"
        "Predict synthetic support ticket priority: low, medium or high.\n\n"
        "## Inputs\n"
        "Text, channel, customer tier, product area, sentiment score and account age.\n\n"
        "## Model\n"
        "TF-IDF for text, one-hot encoding for categories, scaling for numeric features, Logistic Regression classifier.\n\n"
        f"## Metrics\n"
        f"Baseline macro F1: {baseline_metrics['macro_f1']}\n\n"
        f"Final model macro F1: {final_metrics['macro_f1']}\n\n"
        "## Limitations\n"
        "The dataset is synthetic. The model demonstrates an engineering pipeline, not production readiness.\n",
        encoding="utf-8",
    )

    LOGGER.info("Saved model to %s", model_path)
    LOGGER.info("Saved metrics to %s", metrics_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ticket-priority model")
    parser.add_argument("--config", default="configs/config.json", help="Path to JSON config")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    metrics = train_from_config(args.config)
    print(json.dumps({"baseline": metrics["baseline"], "final_model": metrics["final_model"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
