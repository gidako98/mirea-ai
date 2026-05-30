from sklearn.model_selection import train_test_split

from src.data import generate_synthetic_tickets
from src.modeling import build_baseline_model, build_final_model, evaluate_model, split_features_target


def test_final_model_trains_and_beats_minimum_quality():
    data = generate_synthetic_tickets(rows=180, seed=42)
    x, y = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

    baseline = build_baseline_model()
    baseline.fit(x_train, y_train)
    baseline_metrics = evaluate_model(baseline, x_test, y_test)

    model = build_final_model(max_text_features=500, c_value=1.5, max_iter=500)
    model.fit(x_train, y_train)
    final_metrics = evaluate_model(model, x_test, y_test)

    assert final_metrics["macro_f1"] > baseline_metrics["macro_f1"]
    assert final_metrics["accuracy"] >= 0.75
