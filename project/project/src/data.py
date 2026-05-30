"""Synthetic data generation and loading utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "text",
    "channel",
    "customer_tier",
    "product_area",
    "sentiment_score",
    "account_age_days",
]
TARGET_COLUMN = "priority"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

CHANNELS = ["email", "chat", "phone", "web"]
CUSTOMER_TIERS = ["free", "standard", "premium", "enterprise"]
PRODUCT_AREAS = ["billing", "login", "payments", "performance", "security", "integrations"]

LOW_TEXTS = [
    "How can I change my notification settings?",
    "I need help finding the invoice download button.",
    "Question about updating my profile picture.",
    "Where can I see the product documentation?",
    "Minor typo in the help page, no urgent impact.",
]
MEDIUM_TEXTS = [
    "Payment failed for one customer but retry worked later.",
    "Dashboard is slow and several charts time out.",
    "Integration stopped syncing for one workspace.",
    "User cannot reset password after multiple attempts.",
    "Export file is missing some recent records.",
]
HIGH_TEXTS = [
    "Production outage: all users cannot log in.",
    "Security incident with suspicious account access.",
    "Enterprise customer reports data loss after migration.",
    "Payment processing is down for all regions.",
    "Critical API errors affecting premium customers.",
]


def _weighted_choice(rng: np.random.Generator, labels: Iterable[str], probabilities: Iterable[float]) -> str:
    labels_list = list(labels)
    return str(rng.choice(labels_list, p=list(probabilities)))


def generate_synthetic_tickets(rows: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic synthetic dataset for ticket-priority classification."""
    if rows < 30:
        raise ValueError("rows must be at least 30 to keep all classes represented")

    rng = np.random.default_rng(seed)
    records = []

    for i in range(rows):
        priority = _weighted_choice(rng, ["low", "medium", "high"], [0.38, 0.42, 0.20])
        if priority == "low":
            text = str(rng.choice(LOW_TEXTS))
            sentiment = float(np.clip(rng.normal(0.35, 0.25), -1, 1))
            tier_probs = [0.45, 0.35, 0.15, 0.05]
            area_probs = [0.12, 0.16, 0.10, 0.12, 0.10, 0.40]
        elif priority == "medium":
            text = str(rng.choice(MEDIUM_TEXTS))
            sentiment = float(np.clip(rng.normal(-0.10, 0.30), -1, 1))
            tier_probs = [0.20, 0.35, 0.30, 0.15]
            area_probs = [0.20, 0.20, 0.16, 0.22, 0.07, 0.15]
        else:
            text = str(rng.choice(HIGH_TEXTS))
            sentiment = float(np.clip(rng.normal(-0.55, 0.25), -1, 1))
            tier_probs = [0.05, 0.15, 0.30, 0.50]
            area_probs = [0.12, 0.20, 0.20, 0.14, 0.24, 0.10]

        # Add small lexical variation so the model learns more than exact duplicates.
        if rng.random() < 0.22:
            text += " Please respond as soon as possible."
        if rng.random() < 0.15:
            text += " Customer impact is increasing."

        # Add modest label noise to keep the educational task realistic:
        # in real triage data, human labels are not perfectly consistent.
        observed_priority = priority
        if rng.random() < 0.08:
            observed_priority = str(rng.choice([label for label in ["low", "medium", "high"] if label != priority]))

        records.append(
            {
                "ticket_id": f"TCKT-{i + 1:05d}",
                "text": text,
                "channel": str(rng.choice(CHANNELS, p=[0.35, 0.30, 0.15, 0.20])),
                "customer_tier": str(rng.choice(CUSTOMER_TIERS, p=tier_probs)),
                "product_area": str(rng.choice(PRODUCT_AREAS, p=area_probs)),
                "sentiment_score": round(sentiment, 3),
                "account_age_days": int(rng.integers(5, 2200)),
                "priority": observed_priority,
            }
        )

    data = pd.DataFrame.from_records(records)
    return data.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a ticket dataset from CSV."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = pd.read_csv(dataset_path)
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values")
    return data


def save_dataset(data: pd.DataFrame, output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic support-ticket data.")
    parser.add_argument("--output", default="data/sample_tickets.csv", help="Output CSV path")
    parser.add_argument("--rows", type=int, default=600, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = generate_synthetic_tickets(rows=args.rows, seed=args.seed)
    save_dataset(data, args.output)
    print(f"Saved {len(data)} rows to {args.output}")


if __name__ == "__main__":
    main()
