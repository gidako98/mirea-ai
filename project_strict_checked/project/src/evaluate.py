"""Evaluate a saved model on a CSV dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.config import load_config
from src.data import load_dataset
from src.modeling import evaluate_model, split_features_target


def evaluate_saved_model(config_path: str = "configs/config.json") -> dict:
    config = load_config(config_path)
    model_path = Path(config["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run `python -m src.train` first.")
    model = joblib.load(model_path)
    data = load_dataset(config["data_path"])
    x, y = split_features_target(data)
    return evaluate_model(model, x, y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved model")
    parser.add_argument("--config", default="configs/config.json", help="Path to JSON config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_saved_model(args.config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
