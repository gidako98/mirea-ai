# Source code

Main entry points:

- `python -m src.data` — generate synthetic data;
- `python -m src.train` — train baseline and final model, save artifacts;
- `python -m src.evaluate` — evaluate saved model;
- `python -m src.predict` — run CLI inference;
- `python -m src.service` — run HTTP service with `/health` and `/predict`.

The service is implemented using Python standard library `http.server`, not Flask/FastAPI. This keeps the dependency list small while still satisfying the service requirements.
