"""OpenAI-compatible stdlib HTTP server (no external deps).

Endpoints:
  GET  /health                -> {"status": "ok"}
  GET  /v1/models             -> list of models
  POST /v1/video/generations  -> generate video/audio latents for a prompt
  POST /v1/distill            -> run distillation and return metrics
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def create_app(engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "skill": engine.trainer.skill})
            elif self.path == "/v1/models":
                self._send(200, {"data": [{"id": "dmadforge-student", "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/v1/video/generations":
                ln = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(ln) or b"{}")
                prompt = data.get("prompt", "")
                n_frames = int(data.get("n_frames", 4))
                seed = data.get("seed", "http")
                r = engine.generate(prompt, n_frames=n_frames, seed=seed)
                self._send(200, r)
            elif self.path == "/v1/distill":
                ln = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(ln) or b"{}")
                epochs = int(data.get("epochs", engine.cfg.train.epochs))
                r = engine.distill_report(epochs)
                self._send(200, r)
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def run_server(engine, host="127.0.0.1", port=8765):
    handler = create_app(engine)
    httpd = HTTPServer((host, port), handler)
    print("[serve] dmadforge HTTP server on http://%s:%d" % (host, port), flush=True)
    httpd.serve_forever()
