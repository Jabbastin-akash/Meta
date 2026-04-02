"""
Minimal API layer for the Search Ranking Environment.

Exposes HTTP endpoints for OpenEnv interaction and Hugging Face Space
deployment.  Uses only the Python standard library (http.server) so
there are zero extra dependencies beyond pydantic (already required
by the environment).

Endpoints
---------
  GET  /           → health-check / welcome
  GET  /health     → {"status": "ok"}
  POST /reset      → reset environment, return observation JSON
  POST /step       → submit action, return (observation, reward, done, info)
  GET  /state      → return current observation JSON
"""

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Ensure local imports work when run from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import SearchRankingEnv
from models import Action


# ---------------------------------------------------------------------------
# Shared environment instance (single-tenant, stateful)
# ---------------------------------------------------------------------------

ENV = SearchRankingEnv(seed=42)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class EnvHandler(BaseHTTPRequestHandler):
    """Handle HTTP requests for the OpenEnv API."""

    # Silence per-request log lines in production; stderr still works
    def log_message(self, fmt, *args):
        pass  # silent

    # -- helpers ----------------------------------------------------------

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw)

    # -- GET routes -------------------------------------------------------

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/")

        if path in ("", "/"):
            self._send_json({
                "name": "Search Ranking Environment",
                "version": "1.0.0",
                "status": "running",
                "endpoints": ["/", "/health", "/reset", "/step", "/state"],
            })

        elif path == "/health":
            self._send_json({"status": "ok"})

        elif path == "/state":
            obs = ENV.state()
            self._send_json(_observation_to_dict(obs))

        else:
            self._send_json({"error": f"Not found: {path}"}, status=404)

    # -- POST routes ------------------------------------------------------

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")

        if path == "/reset":
            self._handle_reset()
        elif path == "/step":
            self._handle_step()
        else:
            self._send_json({"error": f"Not found: {path}"}, status=404)

    # -- /reset -----------------------------------------------------------

    def _handle_reset(self):
        try:
            body = self._read_body()
            task = body.get("task", "easy")
            obs = ENV.reset(task)
            self._send_json(_observation_to_dict(obs))
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    # -- /step ------------------------------------------------------------

    def _handle_step(self):
        try:
            body = self._read_body()
            ranking = body.get("ranking", [])
            action = Action(ranking=ranking)
            obs, reward, done, info = ENV.step(action)
            self._send_json({
                "observation": _observation_to_dict(obs),
                "reward": reward.score,
                "done": done,
                "info": {
                    "ndcg": info.ndcg,
                    "precision_at_k": info.precision_at_k,
                    "mrr": info.mrr,
                },
            })
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _observation_to_dict(obs) -> dict:
    """Convert an Observation to a JSON-safe dict (excluding internal fields)."""
    return {
        "query": obs.query,
        "documents": [
            {"id": doc.id, "text": doc.text}
            for doc in obs.documents
        ],
    }


# ---------------------------------------------------------------------------
# Server entry-point
# ---------------------------------------------------------------------------

def main():
    port = int(os.environ.get("PORT", "7860"))  # HF Spaces default
    server = HTTPServer(("0.0.0.0", port), EnvHandler)
    print(f"Search Ranking Env API listening on http://0.0.0.0:{port}")
    print(f"  GET  /          → welcome / info")
    print(f"  GET  /health    → health check")
    print(f"  POST /reset     → reset(task)  body: {{\"task\": \"easy\"}}")
    print(f"  POST /step      → step(action) body: {{\"ranking\": [...]}}")
    print(f"  GET  /state     → current observation")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
