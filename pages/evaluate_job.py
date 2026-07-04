#!/usr/bin/env python3
"""DriveRobust Apollo evaluation hook.

This hook is called by local_eval_server.py when the bridge is READY. It uses
Apollo's existing JSONL obstacle publisher inside the Apollo Docker container to
publish a finite small50/scenario sequence onto /apollo/perception/obstacles.

Inputs from local_eval_server.py:
  --job-dir <bridge>/eval_queue/<job_id>
  --archive <uploaded package>
  --result <bridge>/eval_results/<job_id>.json

The unified adversarial settings are read from task_config.json in job-dir.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_CONTAINER = "apollo_neo_dev_11.0.0_pkg"
DEFAULT_CHANNEL = "/apollo/perception/obstacles"
DEFAULT_RATE = 20.0
DEFAULT_TIMEOUT = 120

SMALL50_JSONL = {
    "yolov5s": "/apollo/data/sim_eval/small50/yolo_best_pt_obstacles.jsonl",
    "yolo": "/apollo/data/sim_eval/small50/yolo_best_pt_obstacles.jsonl",
    "yolo_best_pt": "/apollo/data/sim_eval/small50/yolo_best_pt_obstacles.jsonl",
    "avod": "/apollo/data/sim_eval/small50/avod_00175000_obstacles.jsonl",
    "avod_00175000": "/apollo/data/sim_eval/small50/avod_00175000_obstacles.jsonl",
}

SCENARIO_JSONL = {
    "small50": "/apollo/data/sim_eval/small50/yolo_best_pt_obstacles.jsonl",
    "clean_reference_50": "/apollo/data/sim_eval/scenario_suite/subsets/clean_reference_50.jsonl",
    "weather_150": "/apollo/data/sim_eval/scenario_suite/subsets/weather_150.jsonl",
    "lighting_150": "/apollo/data/sim_eval/scenario_suite/subsets/lighting_150.jsonl",
    "sensor_quality_200": "/apollo/data/sim_eval/scenario_suite/subsets/sensor_quality_200.jsonl",
    "occlusion_attack_200": "/apollo/data/sim_eval/scenario_suite/subsets/occlusion_attack_200.jsonl",
    "all_750": "/apollo/data/sim_eval/scenario_suite/subsets/all_750.jsonl",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_result(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(cmd: List[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)


def docker_available(container: str, timeout: int) -> Tuple[bool, Dict[str, Any]]:
    info: Dict[str, Any] = {"container": container}
    try:
        ps = run(["docker", "inspect", container, "--format", "{{.State.Status}}"], timeout=min(timeout, 20))
        info["inspect_returncode"] = ps.returncode
        info["inspect_stdout"] = ps.stdout.strip()
        info["inspect_stderr"] = ps.stderr.strip()[-2000:]
        if ps.returncode != 0:
            return False, info
        if ps.stdout.strip() != "running":
            st = run(["docker", "start", container], timeout=min(timeout, 60))
            info["start_returncode"] = st.returncode
            info["start_stdout"] = st.stdout.strip()
            info["start_stderr"] = st.stderr.strip()[-2000:]
            if st.returncode != 0:
                return False, info
        return True, info
    except Exception as exc:
        info["error"] = str(exc)
        return False, info


def select_jsonl(task_config: Dict[str, Any], manifest: Dict[str, Any]) -> Tuple[str, str, str]:
    dataset_subset = str(task_config.get("dataset_subset") or manifest.get("dataset_subset") or "small50")
    model = str(task_config.get("model") or "yolov5s").lower()
    explicit = os.environ.get("DRIVEROBUST_APOLLO_JSONL", "").strip()
    if explicit:
        return explicit, model, "env:DRIVEROBUST_APOLLO_JSONL"
    if dataset_subset == "small50":
        if model in SMALL50_JSONL:
            return SMALL50_JSONL[model], model, "small50-model-map"
        # BEVFusion currently has no generated Apollo obstacle JSONL in this workspace.
        # Use YOLO small50 as an Apollo-interface fallback but mark it explicitly.
        if model == "bevfusion":
            return SMALL50_JSONL["yolov5s"], model, "small50-bevfusion-fallback-to-yolo-jsonl"
    if dataset_subset in SCENARIO_JSONL:
        return SCENARIO_JSONL[dataset_subset], model, "scenario-subset-map"
    return SMALL50_JSONL["yolov5s"], model, "default-small50-yolo-jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="DriveRobust Apollo Docker evaluation hook")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    result_path = Path(args.result)
    archive = Path(args.archive)
    job = read_json(job_dir / "job.json", {}) or {}
    task_config = read_json(job_dir / "task_config.json", {}) or job.get("task_config") or {}
    manifest = job.get("manifest") or read_json(job_dir / "manifest.json", {}) or {}

    container = os.environ.get("DRIVEROBUST_APOLLO_CONTAINER", DEFAULT_CONTAINER)
    timeout = int(os.environ.get("DRIVEROBUST_APOLLO_TIMEOUT", str(DEFAULT_TIMEOUT)))
    rate = float(os.environ.get("DRIVEROBUST_APOLLO_RATE", str(DEFAULT_RATE)))
    channel = os.environ.get("DRIVEROBUST_APOLLO_CHANNEL", DEFAULT_CHANNEL)
    frame_count = int((manifest or {}).get("frame_count") or 50)
    limit = int(os.environ.get("DRIVEROBUST_APOLLO_LIMIT", str(max(1, min(frame_count, 50)))))
    module_name = os.environ.get("DRIVEROBUST_APOLLO_MODULE", f"driverobust_{job.get('job_id', job_dir.name)}")
    start_dreamview = os.environ.get("DRIVEROBUST_APOLLO_START_DREAMVIEW", "0").lower() in ("1", "true", "yes")

    jsonl, model, selection_reason = select_jsonl(task_config, manifest)
    base_result: Dict[str, Any] = {
        "job_id": job.get("job_id", job_dir.name),
        "status": "running",
        "evaluation_mode": "apollo_cyber_publish",
        "archive": str(archive),
        "manifest": manifest,
        "task_config": task_config,
        "apollo": {
            "container": container,
            "channel": channel,
            "module_name": module_name,
            "jsonl": jsonl,
            "jsonl_selection": selection_reason,
            "requested_model": model,
            "rate": rate,
            "limit": limit,
        },
        "started_at": now_iso(),
    }

    ok, docker_info = docker_available(container, timeout)
    base_result["docker"] = docker_info
    if not ok:
        base_result.update({
            "status": "failed",
            "error": "Apollo Docker container is not available or could not be started",
            "completed_at": now_iso(),
        })
        write_result(result_path, base_result)
        return 2

    shell_parts = ["set -e", "source /opt/apollo/neo/setup.sh >/dev/null 2>&1"]
    if start_dreamview:
        shell_parts.extend([
            "cd /apollo_workspace",
            "aem profile use sample >/tmp/driverobust_aem_profile.log 2>&1 || cat /tmp/driverobust_aem_profile.log",
            "aem bootstrap start --plus >/tmp/driverobust_aem_bootstrap.log 2>&1 || cat /tmp/driverobust_aem_bootstrap.log",
            "tail -5 /tmp/driverobust_aem_bootstrap.log || true",
        ])
    pub_cmd = [
        "python3", "/apollo_workspace/scripts/jsonl_obstacle_publisher.py",
        "--jsonl", jsonl,
        "--channel", channel,
        "--module-name", module_name,
        "--rate", str(rate),
        "--limit", str(limit),
    ]
    shell_parts.append(" ".join(shlex.quote(x) for x in pub_cmd))
    inner = "; ".join(shell_parts)
    cmd = ["docker", "exec", container, "bash", "-lc", inner]
    base_result["apollo"]["command"] = " ".join(shlex.quote(x) for x in cmd)

    try:
        proc = run(cmd, timeout=timeout)
        base_result.update({
            "status": "completed" if proc.returncode == 0 else "failed",
            "apollo_returncode": proc.returncode,
            "apollo_stdout_tail": proc.stdout[-12000:],
            "apollo_stderr_tail": proc.stderr[-12000:],
            "completed_at": now_iso(),
        })
        if proc.returncode == 0:
            base_result["note"] = "Published uploaded/evaluation task through Apollo Cyber perception obstacle channel."
        else:
            base_result["error"] = "Apollo publisher command failed"
        write_result(result_path, base_result)
        return 0 if proc.returncode == 0 else 3
    except subprocess.TimeoutExpired as exc:
        base_result.update({
            "status": "failed",
            "error": f"Apollo publisher timeout after {timeout}s: {exc}",
            "completed_at": now_iso(),
        })
        write_result(result_path, base_result)
        return 4
    except Exception as exc:
        base_result.update({
            "status": "failed",
            "error": str(exc),
            "completed_at": now_iso(),
        })
        write_result(result_path, base_result)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
