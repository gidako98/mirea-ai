from src.data import REQUIRED_COLUMNS, generate_synthetic_tickets


def test_generated_data_has_required_columns_and_classes():
    data = generate_synthetic_tickets(rows=120, seed=7)

    assert set(REQUIRED_COLUMNS).issubset(data.columns)
    assert len(data) == 120
    assert set(data["priority"].unique()) == {"low", "medium", "high"}
    assert data["text"].str.len().min() > 0
