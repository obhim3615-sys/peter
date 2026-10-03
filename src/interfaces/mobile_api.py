"""
Peter Mobile & Antigravity Live Bridge Server (`src/interfaces/mobile_api.py`).

Provides HTTP JSON endpoints for the `peterApp` Android Coding Agent IDE:
  - `/api/mobile/status`: Health & connected model/database info
  - `/api/mobile/ask`: Runs the full Peter LangGraph orchestrator
  - `/api/mobile/save-file`: Saves an edited file from the phone directly to the Django workspace
  - `/api/mobile/git/pull`: Runs `git pull` on the active project and returns updated files
  - `/api/mobile/git/push`: Writes modified files, runs `git add`, `git commit`, and `git push`
  - `/api/mobile/django/run`: Runs `python manage.py <command>` on the Django project
  - `/api/mobile/sync-db`: Re-exports and streams the latest `peter_offline.db` to the phone

Run with:
  python -m src.interfaces.mobile_api
"""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.graph import app as peter_graph
from src.nodes import PRIMARY_MODEL_PROVIDER
from src.utils.logger import logger
from src.utils.project_config import get_active_project_root

MOBILE_DB_PATH = Path(r"C:\Users\Abhishek.Pandey\Videos\peterApp\app\src\main\assets\peter_offline.db")


class PeterMobileRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Google-Account")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Google-Account")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/mobile/status":
            google_email = self.headers.get("X-Google-Account", "Signed-in Google User")
            self._send_json(
                {
                    "status": "online",
                    "agent": "Peter + Antigravity Live Bridge",
                    "model_provider": PRIMARY_MODEL_PROVIDER,
                    "project_root": get_active_project_root(),
                    "authenticated_google_account": google_email,
                    "api_key_required": False,
                }
            )
            return

        if parsed.path == "/api/mobile/sync-db":
            try:
                from scripts.export_mobile_sqlite import build_mobile_sqlite_db

                stats = build_mobile_sqlite_db()
                if MOBILE_DB_PATH.exists():
                    data = MOBILE_DB_PATH.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("X-Peter-Stats", json.dumps(stats))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, status=500)
                return

        self._send_json({"status": "ok", "service": "Peter Mobile API Bridge"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8") if length > 0 else "{}")
        except Exception:
            body = {}

        project_root = Path(get_active_project_root())

        if parsed.path == "/api/mobile/save-file":
            rel_path = (body.get("rel_path") or "").strip().lstrip("/\\")
            content = body.get("content") or ""
            if not rel_path:
                self._send_json({"error": "Missing rel_path"}, status=400)
                return
            target = (project_root / rel_path).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            self._send_json({"status": "saved", "saved_to": str(target)})
            return

        if parsed.path == "/api/mobile/git/pull":
            branch = (body.get("branch") or "main").strip()
            proc = subprocess.run(
                ["git", "pull", "origin", branch],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=30,
            )
            out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            self._send_json({"status": "ok", "output": f"$ git pull origin {branch}\n{out.strip()}", "files": []})
            return

        if parsed.path == "/api/mobile/git/push":
            branch = (body.get("branch") or "main").strip()
            commit_msg = (body.get("commit_message") or "Update from Antigravity Mobile Agent").strip()
            files = body.get("files") or []
            for f in files:
                rp = (f.get("rel_path") or "").strip().lstrip("/\\")
                if rp:
                    dest = (project_root / rp).resolve()
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(f.get("content") or "", encoding="utf-8")

            subprocess.run(["git", "add", "-A"], cwd=str(project_root), capture_output=True, text=True)
            c_proc = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=str(project_root),
                capture_output=True,
                text=True,
            )
            p_proc = subprocess.run(
                ["git", "push", "origin", branch],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=30,
            )
            combined = f"{c_proc.stdout}\n{p_proc.stdout}\n{p_proc.stderr}".strip()
            self._send_json({"status": "ok", "output": f"✅ PC Bridge Git Push ({branch}):\n{combined}"})
            return

        if parsed.path == "/api/mobile/django/run":
            cmd_str = (body.get("command") or "check").strip()
            args = [sys.executable, "manage.py"] + cmd_str.split()
            proc = subprocess.run(
                args,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=25,
            )
            self._send_json(
                {
                    "status": "ok",
                    "stdout": proc.stdout or "",
                    "stderr": proc.stderr or "",
                    "exit_code": proc.returncode,
                }
            )
            return

        if parsed.path != "/api/mobile/ask":
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            question = (body.get("question") or "").strip()
            google_account = (
                body.get("google_account")
                or self.headers.get("X-Google-Account")
                or "Google Account Session"
            )
            offline_context = (body.get("offline_context") or "").strip()

            if not question:
                self._send_json({"error": "Missing 'question' field"}, status=400)
                return

            logger.info(f"[Mobile Online Mode | {google_account}] Question: {question}")
            initial_state = {
                "user_request": question,
                "project_root": str(project_root),
                "history": [],
                "retry_count": 0,
                "prior_execution_summary": offline_context[:3000] if offline_context else "",
            }
            final_state = peter_graph.invoke(initial_state)
            answer = (final_state.get("execution_result") or "No answer produced.").strip()

            self._send_json(
                {
                    "status": "completed",
                    "question": question,
                    "answer": answer,
                    "google_account": google_account,
                    "mode": f"Online ({PRIMARY_MODEL_PROVIDER})",
                }
            )
        except Exception as e:
            logger.error(f"Mobile API error: {traceback.format_exc()}")
            self._send_json({"status": "error", "message": str(e)}, status=500)


def run_mobile_server(host: str = "0.0.0.0", port: int = 8090) -> None:
    server = ThreadingHTTPServer((host, port), PeterMobileRequestHandler)
    logger.info(f"📱 Peter Mobile & Antigravity Online Bridge running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_mobile_server()

