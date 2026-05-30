from src.data import generate_synthetic_tickets
from src.modeling import build_final_model, predict_records, split_features_target


def test_predict_records_returns_label_and_probabilities():
    data = generate_synthetic_tickets(rows=120, seed=10)
    x, y = split_features_target(data)
    model = build_final_model(max_text_features=500, c_value=1.0, max_iter=400)
    model.fit(x, y)

    record = {
        "text": "Security incident with suspicious account access.",
        "channel": "email",
        "customer_tier": "enterprise",
        "product_area": "security",
        "sentiment_score": -0.9,
        "account_age_days": 800,
    }
    result = predict_records(model, [record])

    assert len(result) == 1
    assert result[0]["priority"] in {"low", "medium", "high"}
    assert set(result[0]["probabilities"]) == {"low", "medium", "high"}
