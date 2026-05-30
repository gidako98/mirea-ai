"""Small HTTP service exposing /health and /predict without an extra web framework."""

from __future__ import annotations

import argparse
import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib

from src.config import load_config
from src.modeling import predict_records

LOGGER = logging.getLogger(__name__)
MODEL: Any | None = None
MODEL_PATH: str | None = None


def load_model(model_path: str | Path) -> Any:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def normalize_payload(payload: Any) -> list[dict[str, Any]]:
    """Normalize API payload into a list of records."""
    if isinstance(payload, dict) and "items" in payload:
        payload = payload["items"]
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("All items must be JSON objects")
        return payload
    raise ValueError("Payload must be a JSON object, a list, or {'items': [...]}.")


def make_prediction_response(model: Any, payload: Any) -> dict[str, Any]:
    records = normalize_payload(payload)
    predictions = predict_records(model, records)
    return {"count": len(predictions), "predictions": predictions}


class TicketPriorityHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the model service."""

    server_version = "TicketPriorityService/1.0"

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        response = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self) -> None:  # noqa: N802 - stdlib API name
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model_loaded": MODEL is not None,
                    "model_path": MODEL_PATH,
                    "service": "ticket-priority-triage",
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found", "available_endpoints": ["/health", "/predict"]})

    def do_POST(self) -> None:  # noqa: N802 - stdlib API name
        path = urlparse(self.path).path
        if path != "/predict":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found", "available_endpoints": ["/health", "/predict"]})
            return

        if MODEL is None:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Model is not loaded"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw_body or "{}")
            result = make_prediction_response(MODEL, payload)
            LOGGER.info("prediction_request count=%s client=%s", result["count"], self.client_address[0])
            self._send_json(HTTPStatus.OK, result)
        except json.JSONDecodeError:
            LOGGER.warning("invalid_json client=%s", self.client_address[0])
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Request body must be valid JSON"})
        except ValueError as exc:
            LOGGER.warning("bad_request error=%s client=%s", exc, self.client_address[0])
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive branch
            LOGGER.exception("prediction_failed")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Prediction failed", "detail": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("http_access " + format, *args)


def run_service(config_path: str = "configs/config.json") -> None:
    """Load the model and run the HTTP service."""
    global MODEL, MODEL_PATH

    config = load_config(config_path)
    model_path = os.getenv("AIE_MODEL_PATH", config["model_path"])
    host = os.getenv("AIE_SERVICE_HOST", config["service"]["host"])
    port = int(os.getenv("AIE_SERVICE_PORT", config["service"]["port"]))

    MODEL_PATH = str(model_path)
    MODEL = load_model(model_path)

    server = ThreadingHTTPServer((host, port), TicketPriorityHandler)
    LOGGER.info("service_started host=%s port=%s model_path=%s", host, port, model_path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("service_stopped_by_user")
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ticket priority HTTP service")
    parser.add_argument("--config", default="configs/config.json", help="Path to JSON config")
    return parser.parse_args()


def main() -> None:
    level_name = os.getenv("AIE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    run_service(args.config)


if __name__ == "__main__":
    main()
