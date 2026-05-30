"""CLI inference for the trained ticket-priority model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import load_config
from src.modeling import predict_records


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load one record or a list of records from a JSON file."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict) and "items" in payload:
        payload = payload["items"]
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError("Input JSON must be an object, a list of objects, or {'items': [...]}.")


def predict_file(input_path: str | Path, config_path: str | Path = "configs/config.json", output: str | Path | None = None) -> list[dict[str, Any]]:
    config = load_config(config_path)
    model_path = Path(config["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run `python -m src.train` first.")
    model = joblib.load(model_path)
    records = load_records(input_path)
    predictions = predict_records(model, records)

    if output is not None:
        rows = []
        for source, prediction in zip(records, predictions):
            row = dict(source)
            row["predicted_priority"] = prediction["priority"]
            for key, value in prediction.get("probabilities", {}).items():
                row[f"probability_{key}"] = value
            rows.append(row)
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ticket priority from JSON input")
    parser.add_argument("--input", required=True, help="Path to input JSON")
    parser.add_argument("--config", default="configs/config.json", help="Path to JSON config")
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_file(args.input, args.config, args.output)
    print(json.dumps({"predictions": predictions}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
