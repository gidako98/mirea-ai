# Artifacts

Generated project artifacts are stored here:

- `model.joblib` — trained scikit-learn pipeline used by CLI and HTTP service;
- `metrics.json` — comparison of baseline and final model metrics;
- `model_card.md` — short description of model purpose, inputs, metrics and limitations;
- `sample_predictions.csv` — example predictions created by `src.predict`.

Regenerate artifacts:

```bash
python -m src.train --config configs/config.json
python -m src.predict --input examples/sample_input.json --output artifacts/sample_predictions.csv
```
