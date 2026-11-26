"""Local educational inference endpoint; no real transaction data."""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from model import score


def make_handler(model):
    lock = threading.Lock()
    counters = {"requests": 0, "invalid": 0, "seconds": 0.0}

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload, content_type="application/json"):
            data = json.dumps(payload).encode() if content_type == "application/json" else payload.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/healthz":
                self.respond(200, {"status": "ok", "model": model["model_version"]})
            elif self.path == "/metrics":
                with lock:
                    text = f'fraud_predictions_total {counters["requests"]}\nfraud_invalid_requests_total {counters["invalid"]}\nfraud_prediction_seconds_sum {counters["seconds"]}\n'
                self.respond(200, text, "text/plain; version=0.0.4")
            else:
                self.respond(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                self.respond(404, {"error": "not found"})
                return
            started = time.monotonic()
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("invalid body size")
                payload = json.loads(self.rfile.read(size))
                value = score(model, payload["features"])
            except (ValueError, TypeError, KeyError, UnicodeError):
                with lock:
                    counters["invalid"] += 1
                self.respond(400, {"error": "expected three normalized synthetic numeric features"})
                return
            with lock:
                counters["requests"] += 1
                counters["seconds"] += time.monotonic() - started
            self.respond(200, {"score": round(value, 6), "model": model["model_version"], "synthetic": True})

    return Handler


if __name__ == "__main__":
    model = json.loads(Path("model.json").read_text(encoding="utf-8"))
    ThreadingHTTPServer((os.getenv("HOST", "127.0.0.1"), 8080), make_handler(model)).serve_forever()
