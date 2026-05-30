from src.data import generate_synthetic_tickets
from src.modeling import build_final_model, split_features_target
from src.service import make_prediction_response, normalize_payload


def test_service_payload_normalization_single_and_batch():
    single = {"text": "x"}
    batch = {"items": [{"text": "x"}, {"text": "y"}]}

    assert normalize_payload(single) == [single]
    assert normalize_payload(batch) == batch["items"]


def test_make_prediction_response_uses_real_model():
    data = generate_synthetic_tickets(rows=120, seed=33)
    x, y = split_features_target(data)
    model = build_final_model(max_text_features=500, c_value=1.0, max_iter=400)
    model.fit(x, y)

    payload = {
        "text": "Payment processing is down for all regions.",
        "channel": "chat",
        "customer_tier": "premium",
        "product_area": "payments",
        "sentiment_score": -0.7,
        "account_age_days": 400,
    }
    response = make_prediction_response(model, payload)

    assert response["count"] == 1
    assert response["predictions"][0]["priority"] in {"low", "medium", "high"}
