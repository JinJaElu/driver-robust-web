#!/usr/bin/env python3
"""
DriveRobust local evaluation upload gateway.

Run from this folder:
  python3 local_eval_server.py --host 0.0.0.0 --port 8765

The server only writes below this pages/ directory:
  runtime/, incoming_uploads/, eval_queue/, eval_results/
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "runtime"
UPLOAD_DIR = BASE_DIR / "incoming_uploads"
QUEUE_DIR = BASE_DIR / "eval_queue"
RESULT_DIR = BASE_DIR / "eval_results"
READY_MARKER = RUNTIME_DIR / "SIM_READY"
SPEC_VERSION = "driverobust-eval-upload-v1"
MAX_UPLOAD_BYTES = int(os.environ.get("DRIVEROBUST_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
WORKER_INTERVAL = float(os.environ.get("DRIVEROBUST_WORKER_INTERVAL", "3"))
UPLOAD_TOKEN = os.environ.get("DRIVEROBUST_UPLOAD_TOKEN", "").strip()

SAFE_NAME = re.compile(r"[^A-Za-z0-9._@+=,-]+")
JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")

for d in (RUNTIME_DIR, UPLOAD_DIR, QUEUE_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def safe_filename(name: str) -> str:
    name = urllib.parse.unquote(name or "upload.zip").split("/")[-1].split("\\")[-1]
    name = SAFE_NAME.sub("_", name).strip("._") or "upload.zip"
    return name[:160]


def make_job_id(candidate: Optional[str] = None) -> str:
    if candidate and JOB_ID_RE.match(candidate):
        return candidate
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"job-{stamp}-{uuid.uuid4().hex[:8]}"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def is_safe_archive_name(name: str) -> bool:
    p = Path(name)
    return not (p.is_absolute() or ".." in p.parts or name.startswith(("/", "\\")))


def detect_archive(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith((".tar.gz", ".tgz", ".tar")):
        return "tar"
    if lower.endswith((".json", ".jsonl")):
        return "single-json"
    return "unknown"


def validate_manifest_obj(obj: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if obj.get("spec_version") != SPEC_VERSION:
        errors.append(f"manifest.spec_version must be {SPEC_VERSION!r}")
    for key in ("task_id", "dataset_type", "sensor_suite", "frame_count", "files"):
        if key not in obj:
            errors.append(f"manifest.{key} is required")
    if "dataset_type" in obj and obj["dataset_type"] not in ("kitti", "nuscenes", "custom"):
        errors.append("manifest.dataset_type must be one of: kitti, nuscenes, custom")
    if "sensor_suite" in obj and not isinstance(obj["sensor_suite"], list):
        errors.append("manifest.sensor_suite must be an array")
    if "frame_count" in obj:
        try:
            if int(obj["frame_count"]) <= 0:
                errors.append("manifest.frame_count must be positive")
        except Exception:
            errors.append("manifest.frame_count must be an integer")
    if "files" in obj and not isinstance(obj["files"], dict):
        errors.append("manifest.files must be an object")
    return errors


def validate_archive(path: Path) -> Tuple[Optional[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Return manifest, errors, stats."""
    kind = detect_archive(path)
    errors: List[str] = []
    stats: Dict[str, Any] = {"archive_type": kind, "member_count": 0, "manifest_found": False}
    manifest: Optional[Dict[str, Any]] = None

    try:
        if kind == "zip":
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
                if bad:
                    errors.append(f"zip CRC check failed at {bad}")
                names = zf.namelist()
                stats["member_count"] = len(names)
                unsafe = [n for n in names if not is_safe_archive_name(n)]
                if unsafe:
                    errors.append(f"unsafe archive paths: {unsafe[:5]}")
                if "manifest.json" in names:
                    stats["manifest_found"] = True
                    manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                else:
                    errors.append("manifest.json must exist at archive root")
        elif kind == "tar":
            with tarfile.open(path) as tf:
                members = tf.getmembers()
                stats["member_count"] = len(members)
                unsafe = [m.name for m in members if not is_safe_archive_name(m.name)]
                if unsafe:
                    errors.append(f"unsafe archive paths: {unsafe[:5]}")
                found = None
                for m in members:
                    if m.name == "manifest.json":
                        found = m
                        break
                if found:
                    stats["manifest_found"] = True
                    f = tf.extractfile(found)
                    if f:
                        manifest = json.loads(f.read().decode("utf-8"))
                else:
                    errors.append("manifest.json must exist at archive root")
        elif kind == "single-json":
            stats["member_count"] = 1
            if path.name.lower().endswith(".json"):
                manifest = json.loads(path.read_text(encoding="utf-8"))
                stats["manifest_found"] = True
            else:
                errors.append("single .jsonl uploads are accepted only as attachments inside a zip/tar package")
        else:
            errors.append("unsupported archive type; use .zip, .tar.gz, .tgz, .tar or manifest .json")
    except Exception as exc:
        errors.append(f"archive/manifest parse error: {exc}")

    if manifest is not None:
        if not isinstance(manifest, dict):
            errors.append("manifest.json must be a JSON object")
        else:
            errors.extend(validate_manifest_obj(manifest))
    return manifest, errors, stats


def simulation_ready() -> bool:
    return os.environ.get("DRIVEROBUST_SIM_READY", "").lower() in ("1", "true", "yes", "ready") or READY_MARKER.exists()


def runner_path() -> Optional[Path]:
    for rel in ("runtime/evaluate_job.py", "runtime/evaluate_job.sh", "evaluate_job.py", "evaluate_job.sh"):
        p = BASE_DIR / rel
        if p.exists():
            return p
    return None


def runner_command(runner: Path, job_dir: Path, archive: Path, result_path: Path) -> List[str]:
    if runner.suffix == ".py":
        return [sys.executable, str(runner), "--job-dir", str(job_dir), "--archive", str(archive), "--result", str(result_path)]
    return ["bash", str(runner), "--job-dir", str(job_dir), "--archive", str(archive), "--result", str(result_path)]


def post_callback(callback_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        callback_url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "DriveRobustLocalEval/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return {"ok": True, "status": resp.status, "response": resp.read(2048).decode("utf-8", errors="replace")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


processing_lock = threading.Lock()
stop_event = threading.Event()


def process_job(job_id: str) -> None:
    with processing_lock:
        job_dir = QUEUE_DIR / job_id
        meta_path = job_dir / "job.json"
        job = read_json(meta_path, {})
        if not job or job.get("status") not in ("queued", "retry"):
            return
        job["status"] = "running"
        job["started_at"] = now_iso()
        write_json(meta_path, job)

        result_path = RESULT_DIR / f"{job_id}.json"
        archive = Path(job["archive_path"])
        runner = runner_path()
        result: Dict[str, Any]
        try:
            if runner:
                cmd = runner_command(runner, job_dir, archive, result_path)
                proc = subprocess.run(cmd, cwd=str(BASE_DIR), text=True, capture_output=True, timeout=int(os.environ.get("DRIVEROBUST_EVAL_TIMEOUT", "7200")))
                result = read_json(result_path, {}) if result_path.exists() else {}
                result.update({
                    "job_id": job_id,
                    "status": "completed" if proc.returncode == 0 else "failed",
                    "runner": str(runner.relative_to(BASE_DIR)),
                    "runner_returncode": proc.returncode,
                    "runner_stdout_tail": proc.stdout[-8000:],
                    "runner_stderr_tail": proc.stderr[-8000:],
                    "completed_at": now_iso(),
                })
            else:
                # Safe built-in fallback: validates and records the package, but does not claim closed-loop simulation.
                result = {
                    "job_id": job_id,
                    "status": "completed",
                    "evaluation_mode": "validation_only",
                    "note": "Simulation marker is ready but no runtime/evaluate_job.py or .sh hook was found; archive validation metadata returned.",
                    "manifest": job.get("manifest"),
                    "archive_stats": job.get("archive_stats"),
                    "completed_at": now_iso(),
                }
            write_json(result_path, result)
            job["status"] = result.get("status", "completed")
            job["result_path"] = str(result_path.relative_to(BASE_DIR))
            job["completed_at"] = result.get("completed_at", now_iso())
        except subprocess.TimeoutExpired as exc:
            result = {"job_id": job_id, "status": "failed", "error": f"evaluation timeout: {exc}", "completed_at": now_iso()}
            write_json(result_path, result)
            job.update({"status": "failed", "error": result["error"], "result_path": str(result_path.relative_to(BASE_DIR)), "completed_at": result["completed_at"]})
        except Exception as exc:
            result = {"job_id": job_id, "status": "failed", "error": str(exc), "completed_at": now_iso()}
            write_json(result_path, result)
            job.update({"status": "failed", "error": str(exc), "result_path": str(result_path.relative_to(BASE_DIR)), "completed_at": result["completed_at"]})

        callback_url = job.get("callback_url") or (job.get("manifest") or {}).get("callback_url")
        if callback_url:
            cb = post_callback(callback_url, result)
            job["callback"] = {"url": callback_url, **cb, "sent_at": now_iso()}
        write_json(meta_path, job)


def worker_loop() -> None:
    while not stop_event.is_set():
        if simulation_ready():
            queued = sorted(QUEUE_DIR.glob("*/job.json"), key=lambda p: p.stat().st_mtime)
            for meta_path in queued:
                job = read_json(meta_path, {})
                if job.get("status") in ("queued", "retry"):
                    process_job(meta_path.parent.name)
                    break
        stop_event.wait(WORKER_INTERVAL)


def list_jobs() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in sorted(QUEUE_DIR.glob("*/job.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        j = read_json(p, {})
        if j:
            items.append(j)
    return items


class Handler(SimpleHTTPRequestHandler):
    server_version = "DriveRobustLocalEval/1.0"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-File-Name, X-Callback-Url, X-Submitter, X-Job-Id, X-Upload-Token, Authorization")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def send_json(self, code: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def auth_ok(self) -> bool:
        if not UPLOAD_TOKEN:
            return True
        auth = self.headers.get("Authorization", "")
        token = self.headers.get("X-Upload-Token", "")
        return token == UPLOAD_TOKEN or auth == f"Bearer {UPLOAD_TOKEN}"

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/api/health":
            jobs = list_jobs()
            queued = sum(1 for j in jobs if j.get("status") in ("queued", "retry"))
            self.send_json(200, {
                "service": "DriveRobust local evaluation gateway",
                "status": "ready" if simulation_ready() else "queueing",
                "simulation_ready": simulation_ready(),
                "ready_marker": str(READY_MARKER.relative_to(BASE_DIR)),
                "runner": str(runner_path().relative_to(BASE_DIR)) if runner_path() else None,
                "queue_depth": queued,
                "spec_version": SPEC_VERSION,
                "upload_endpoint": "/api/evaluate/upload",
                "max_upload_bytes": MAX_UPLOAD_BYTES,
                "time": now_iso(),
            })
            return
        if path == "/api/spec":
            self.send_json(200, upload_spec())
            return
        if path == "/api/tasks":
            self.send_json(200, {"jobs": list_jobs()})
            return
        if path.startswith("/api/tasks/"):
            job_id = path.split("/")[-1]
            job = read_json(QUEUE_DIR / job_id / "job.json")
            if not job:
                self.send_json(404, {"error": "job not found"})
            else:
                self.send_json(200, job)
            return
        if path.startswith("/api/results/"):
            job_id = path.split("/")[-1]
            result = read_json(RESULT_DIR / f"{job_id}.json")
            if not result:
                self.send_json(404, {"error": "result not found or not completed"})
            else:
                self.send_json(200, result)
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path != "/api/evaluate/upload":
            self.send_json(404, {"error": "unknown endpoint"})
            return
        if not self.auth_ok():
            self.send_json(401, {"error": "upload token required or invalid"})
            return
        length_s = self.headers.get("Content-Length")
        if not length_s:
            self.send_json(411, {"error": "Content-Length required"})
            return
        try:
            length = int(length_s)
        except ValueError:
            self.send_json(400, {"error": "invalid Content-Length"})
            return
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self.send_json(413, {"error": "upload too large or empty", "max_upload_bytes": MAX_UPLOAD_BYTES})
            return

        qs = urllib.parse.parse_qs(parsed.query)
        job_id = make_job_id(self.headers.get("X-Job-Id") or (qs.get("job_id") or [None])[0])
        filename = safe_filename(self.headers.get("X-File-Name") or (qs.get("filename") or ["upload.zip"])[0])
        callback_url = self.headers.get("X-Callback-Url") or (qs.get("callback_url") or [""])[0]
        submitter = self.headers.get("X-Submitter") or (qs.get("submitter") or [""])[0]

        job_dir = QUEUE_DIR / job_id
        if job_dir.exists():
            self.send_json(409, {"error": "job_id already exists", "job_id": job_id})
            return
        job_dir.mkdir(parents=True, exist_ok=False)
        archive_path = job_dir / filename
        h = hashlib.sha256()
        remaining = length
        with archive_path.open("wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                remaining -= len(chunk)
        if remaining != 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            self.send_json(400, {"error": "upload stream ended early"})
            return

        manifest, errors, stats = validate_archive(archive_path)
        created = now_iso()
        job = {
            "job_id": job_id,
            "status": "rejected" if errors else ("queued" if not simulation_ready() else "queued"),
            "reason": None if not errors else "invalid_upload_package",
            "queue_note": None if simulation_ready() else "simulation environment is not ready; job is accepted and waiting in queue",
            "created_at": created,
            "updated_at": created,
            "submitter": submitter,
            "callback_url": callback_url,
            "archive_name": filename,
            "archive_path": str(archive_path),
            "archive_bytes": length,
            "sha256": h.hexdigest(),
            "manifest": manifest,
            "archive_stats": stats,
            "validation_errors": errors,
        }
        write_json(job_dir / "job.json", job)
        if errors:
            self.send_json(400, job)
        else:
            self.send_json(202, job)


def upload_spec() -> Dict[str, Any]:
    return {
        "spec_version": SPEC_VERSION,
        "endpoint": "POST /api/evaluate/upload",
        "body": "raw binary archive (.zip/.tar.gz/.tgz/.tar) or manifest .json for dry-run validation",
        "required_headers": ["X-File-Name"],
        "optional_headers": ["X-Job-Id", "X-Callback-Url", "X-Submitter", "X-Upload-Token", "Authorization: Bearer <token>"],
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "manifest_root_file": "manifest.json",
        "manifest_schema": {
            "spec_version": SPEC_VERSION,
            "task_id": "string; stable task id from remote server",
            "dataset_type": "kitti | nuscenes | custom",
            "sensor_suite": ["camera", "lidar"],
            "frame_count": "positive integer",
            "files": {
                "images": "inputs/images/ or samples/<CAM_NAME>/",
                "lidar": "inputs/lidar/ or velodyne/ (optional)",
                "calib": "calib/ (optional)",
                "labels": "labels/ or annotations/ (optional)"
            },
            "callback_url": "optional HTTPS URL; result JSON will be POSTed here",
        },
        "status_rule": "If runtime/SIM_READY or DRIVEROBUST_SIM_READY=1 exists, queued jobs are evaluated; otherwise accepted jobs remain queued.",
        "result_endpoints": ["GET /api/tasks/<job_id>", "GET /api/results/<job_id>"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    worker = threading.Thread(target=worker_loop, name="eval-worker", daemon=True)
    worker.start()
    server = ThreadingHTTPServer((args.host, args.port), lambda *a, **kw: Handler(*a, directory=str(BASE_DIR), **kw))
    print(f"DriveRobust local eval gateway: http://{args.host}:{args.port}/dashboard.html", flush=True)
    print(f"Upload endpoint: http://{args.host}:{args.port}/api/evaluate/upload", flush=True)
    print(f"Ready marker: {READY_MARKER}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
