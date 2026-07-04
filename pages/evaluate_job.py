#!/usr/bin/env python3
"""DriveRobust YOLO-first evaluation hook for uploaded image data.

local_eval_server.py calls this file when the bridge is READY. The expected
upload input is a single image or an archive containing images. JSON is produced
as the evaluation result; JSON/JSONL is not the primary upload input.

Implemented path:
  uploaded images -> YOLOv5 inference -> JSON metrics/result
Optional interface path:
  YOLO detections -> pseudo Apollo obstacle JSONL -> /apollo/perception/obstacles
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

DEFAULT_YOLO_ROOT = "/home/matrix/yolov5"
DEFAULT_YOLO_PYTHON = "/home/matrix/miniconda3/envs/yolov5/bin/python"
DEFAULT_YOLO_WEIGHTS = "/home/matrix/yolov5/yolov5s.pt"
DEFAULT_APOLLO_CONTAINER = "apollo_neo_dev_11.0.0_pkg"
DEFAULT_APOLLO_CHANNEL = "/apollo/perception/obstacles"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def is_image_name(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTS)


def safe_member_name(name: str) -> bool:
    p = Path(name)
    return not (p.is_absolute() or ".." in p.parts or name.startswith(("/", "\\")))


def flat_name(original: str, index: int) -> str:
    p = Path(original)
    suffix = p.suffix.lower() if p.suffix else ".jpg"
    stem = "__".join(part for part in p.with_suffix("").parts if part not in (".", ""))
    safe = "".join(ch if ch.isalnum() or ch in "._-+=" else "_" for ch in stem).strip("._")
    return f"{index:06d}_{safe or 'image'}{suffix}"


def prepare_images(upload: Path, job_dir: Path) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Copy image inputs into a flat image directory for YOLOv5 detect.py."""
    image_dir = job_dir / "input_images_flat"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    images: List[Dict[str, str]] = []
    stats: Dict[str, Any] = {"input_type": "unknown", "image_dir": str(image_dir)}

    def add_bytes(original: str, data: bytes) -> None:
        out = image_dir / flat_name(original, len(images) + 1)
        out.write_bytes(data)
        images.append({"original": original, "path": str(out), "filename": out.name})

    lower = upload.name.lower()
    if is_image_name(lower):
        out = image_dir / flat_name(upload.name, 1)
        shutil.copy2(upload, out)
        images.append({"original": upload.name, "path": str(out), "filename": out.name})
        stats["input_type"] = "single_image"
    elif lower.endswith(".zip"):
        stats["input_type"] = "zip"
        with zipfile.ZipFile(upload) as zf:
            for name in zf.namelist():
                if name.endswith("/") or not is_image_name(name):
                    continue
                if not safe_member_name(name):
                    raise ValueError(f"unsafe archive path: {name}")
                add_bytes(name, zf.read(name))
    elif lower.endswith((".tar.gz", ".tgz", ".tar")):
        stats["input_type"] = "tar"
        with tarfile.open(upload) as tf:
            for member in tf.getmembers():
                if not member.isfile() or not is_image_name(member.name):
                    continue
                if not safe_member_name(member.name):
                    raise ValueError(f"unsafe archive path: {member.name}")
                f = tf.extractfile(member)
                if f:
                    add_bytes(member.name, f.read())
    else:
        raise ValueError("upload must be an image file or image archive")

    if not images:
        raise ValueError("no images found in upload")
    stats["image_count"] = len(images)
    stats["image_examples"] = [x["original"] for x in images[:8]]
    return images, stats


def image_size(path: Path) -> Tuple[int, int]:
    if Image is None:
        return 0, 0
    try:
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0


def run_yolo(image_dir: Path, job_dir: Path, task_config: Dict[str, Any]) -> Tuple[Dict[str, Any], subprocess.CompletedProcess[str]]:
    yolo_python = os.environ.get("DRIVEROBUST_YOLO_PYTHON", DEFAULT_YOLO_PYTHON)
    yolo_root = Path(os.environ.get("DRIVEROBUST_YOLO_ROOT", DEFAULT_YOLO_ROOT))
    weights = os.environ.get("DRIVEROBUST_YOLO_WEIGHTS", DEFAULT_YOLO_WEIGHTS)
    device = os.environ.get("DRIVEROBUST_YOLO_DEVICE", "cpu")
    imgsz = str(os.environ.get("DRIVEROBUST_YOLO_IMGSZ", "640"))
    conf = str(os.environ.get("DRIVEROBUST_YOLO_CONF", "0.25"))
    iou = str(os.environ.get("DRIVEROBUST_YOLO_IOU", "0.45"))
    timeout = int(os.environ.get("DRIVEROBUST_YOLO_TIMEOUT", "900"))
    project = job_dir / "yolo_output"
    name = "predict"
    cmd = [
        yolo_python,
        str(yolo_root / "detect.py"),
        "--weights", weights,
        "--source", str(image_dir),
        "--imgsz", imgsz,
        "--conf-thres", conf,
        "--iou-thres", iou,
        "--device", device,
        "--save-txt",
        "--save-conf",
        "--nosave",
        "--project", str(project),
        "--name", name,
        "--exist-ok",
    ]
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(yolo_root), text=True, capture_output=True, timeout=timeout)
    elapsed = time.time() - start
    meta = {
        "python": yolo_python,
        "root": str(yolo_root),
        "weights": weights,
        "device": device,
        "imgsz": imgsz,
        "conf_thres": conf,
        "iou_thres": iou,
        "output_dir": str(project / name),
        "labels_dir": str(project / name / "labels"),
        "elapsed_sec": round(elapsed, 3),
        "command": " ".join(cmd),
        "requested_model": task_config.get("model", "yolov5s"),
    }
    return meta, proc


def parse_yolo_outputs(images: List[Dict[str, str]], labels_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    frames: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = {}
    detections_total = 0
    conf_sum = 0.0
    apollo_frames: List[Dict[str, Any]] = []

    for seq, item in enumerate(images, 1):
        path = Path(item["path"])
        w, h = image_size(path)
        label_path = labels_dir / f"{path.stem}.txt"
        detections: List[Dict[str, Any]] = []
        obstacles: List[Dict[str, Any]] = []
        if label_path.exists():
            for det_idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                conf = float(parts[5]) if len(parts) >= 6 else None
                name = COCO80[cls_id] if 0 <= cls_id < len(COCO80) else str(cls_id)
                x1 = (xc - bw / 2) * w if w else None
                y1 = (yc - bh / 2) * h if h else None
                x2 = (xc + bw / 2) * w if w else None
                y2 = (yc + bh / 2) * h if h else None
                d = {
                    "class_id": cls_id,
                    "class_name": name,
                    "confidence": conf,
                    "bbox_xywhn": [xc, yc, bw, bh],
                    "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)] if w and h else None,
                }
                detections.append(d)
                detections_total += 1
                class_counts[name] = class_counts.get(name, 0) + 1
                if conf is not None:
                    conf_sum += conf
                # Pseudo-3D deterministic adapter for Apollo interface demonstration.
                apollo_type = "PEDESTRIAN" if name == "person" else ("BICYCLE" if name in ("bicycle", "motorcycle") else "VEHICLE")
                obstacles.append({
                    "id": seq * 1000 + det_idx,
                    "class": name,
                    "type": apollo_type,
                    "sub_type": "ST_PEDESTRIAN" if name == "person" else ("ST_CYCLIST" if apollo_type == "BICYCLE" else "ST_CAR"),
                    "confidence": conf if conf is not None else 0.5,
                    "position": {"x": round(max(1.0, 70.0 * (1.0 - yc)), 3), "y": round((xc - 0.5) * 22.0, 3), "z": 0.0},
                    "dimensions": {"length": 4.2 if apollo_type == "VEHICLE" else 0.8, "width": 1.8 if apollo_type == "VEHICLE" else 0.6, "height": 1.6 if apollo_type == "VEHICLE" else 1.7},
                    "theta": 0.0,
                    "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "bbox2d": {"xmin": round(x1 or 0, 2), "ymin": round(y1 or 0, 2), "xmax": round(x2 or 0, 2), "ymax": round(y2 or 0, 2)},
                    "note": "Pseudo-3D from YOLO 2D bbox for Apollo interface publishing",
                })
        frame = {
            "frame_id": path.stem,
            "source_name": item["original"],
            "image_path": str(path),
            "width": w,
            "height": h,
            "detections": detections,
        }
        frames.append(frame)
        apollo_frames.append({"frame_id": path.stem, "sequence_num": seq, "model": "yolov5", "source": "uploaded_image_yolo", "obstacles": obstacles})

    summary = {
        "images_evaluated": len(images),
        "detections_total": detections_total,
        "images_with_detections": sum(1 for f in frames if f["detections"]),
        "class_counts": class_counts,
        "avg_confidence": round(conf_sum / detections_total, 4) if detections_total else 0.0,
    }
    return frames, summary, apollo_frames


def write_apollo_jsonl(path: Path, frames: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr, ensure_ascii=False) + "\n")


def publish_to_apollo(jsonl: Path, job_id: str, limit: int) -> Dict[str, Any]:
    if os.environ.get("DRIVEROBUST_APOLLO_PUBLISH", "1").lower() not in ("1", "true", "yes"):
        return {"enabled": False, "reason": "DRIVEROBUST_APOLLO_PUBLISH disabled"}
    container = os.environ.get("DRIVEROBUST_APOLLO_CONTAINER", DEFAULT_APOLLO_CONTAINER)
    channel = os.environ.get("DRIVEROBUST_APOLLO_CHANNEL", DEFAULT_APOLLO_CHANNEL)
    rate = os.environ.get("DRIVEROBUST_APOLLO_RATE", "20")
    timeout = int(os.environ.get("DRIVEROBUST_APOLLO_TIMEOUT", "120"))
    module_name = os.environ.get("DRIVEROBUST_APOLLO_MODULE", f"driverobust_yolo_{job_id}")
    container_jsonl = f"/tmp/driverobust_{job_id}_yolo_obstacles.jsonl".replace("/", "_")
    container_jsonl = "/tmp/" + container_jsonl.lstrip("_")
    info: Dict[str, Any] = {"enabled": True, "container": container, "channel": channel, "jsonl": str(jsonl), "container_jsonl": container_jsonl, "module_name": module_name, "limit": limit}
    inspect = subprocess.run(["docker", "inspect", container, "--format", "{{.State.Status}}"], text=True, capture_output=True, timeout=30)
    info["inspect_returncode"] = inspect.returncode
    info["inspect_stdout"] = inspect.stdout.strip()
    info["inspect_stderr"] = inspect.stderr[-2000:]
    if inspect.returncode != 0:
        info.update({"status": "failed", "error": "Apollo container not found"})
        return info
    if inspect.stdout.strip() != "running":
        start = subprocess.run(["docker", "start", container], text=True, capture_output=True, timeout=60)
        info["start_returncode"] = start.returncode
        info["start_stdout"] = start.stdout.strip()
        info["start_stderr"] = start.stderr[-2000:]
        if start.returncode != 0:
            info.update({"status": "failed", "error": "Apollo container could not be started"})
            return info
    cp = subprocess.run(["docker", "cp", str(jsonl), f"{container}:{container_jsonl}"], text=True, capture_output=True, timeout=60)
    info["copy_returncode"] = cp.returncode
    info["copy_stderr"] = cp.stderr[-2000:]
    if cp.returncode != 0:
        info.update({"status": "failed", "error": "docker cp failed"})
        return info
    inner = (
        "set -e; source /opt/apollo/neo/setup.sh >/dev/null 2>&1; "
        f"python3 /apollo_workspace/scripts/jsonl_obstacle_publisher.py --jsonl {container_jsonl} "
        f"--channel {channel} --module-name {module_name} --rate {rate} --limit {max(1, limit)}"
    )
    cmd = ["docker", "exec", container, "bash", "-lc", inner]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    info.update({
        "status": "completed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-8000:],
        "stderr_tail": proc.stderr[-8000:],
        "command": " ".join(cmd),
    })
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="DriveRobust YOLO image evaluation hook")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    archive = Path(args.archive)
    result_path = Path(args.result)
    job = read_json(job_dir / "job.json", {}) or {}
    task_config = read_json(job_dir / "task_config.json", {}) or job.get("task_config") or {}
    manifest = job.get("manifest") or {}
    started = now_iso()

    try:
        images, input_stats = prepare_images(archive, job_dir)
        yolo_meta, proc = run_yolo(Path(input_stats["image_dir"]), job_dir, task_config)
        if proc.returncode != 0:
            result = {
                "job_id": job.get("job_id", job_dir.name),
                "status": "failed",
                "evaluation_mode": "yolov5_image_inference",
                "error": "YOLOv5 inference command failed",
                "started_at": started,
                "completed_at": now_iso(),
                "manifest": manifest,
                "task_config": task_config,
                "input": input_stats,
                "yolo": {**yolo_meta, "returncode": proc.returncode, "stdout_tail": proc.stdout[-8000:], "stderr_tail": proc.stderr[-8000:]},
            }
            write_json(result_path, result)
            return 2
        frames, summary, apollo_frames = parse_yolo_outputs(images, Path(yolo_meta["labels_dir"]))
        apollo_jsonl = job_dir / "yolo_apollo_obstacles.jsonl"
        write_apollo_jsonl(apollo_jsonl, apollo_frames)
        apollo = publish_to_apollo(apollo_jsonl, str(job.get("job_id", job_dir.name)), len(apollo_frames))
        status = "completed" if apollo.get("status") in ("completed", None) or not apollo.get("enabled", True) else "completed_with_apollo_warning"
        result = {
            "job_id": job.get("job_id", job_dir.name),
            "status": status,
            "evaluation_mode": "yolov5_image_inference",
            "started_at": started,
            "completed_at": now_iso(),
            "manifest": manifest,
            "task_config": task_config,
            "input": input_stats,
            "summary": summary,
            "frames": frames,
            "yolo": {**yolo_meta, "returncode": proc.returncode, "stdout_tail": proc.stdout[-8000:], "stderr_tail": proc.stderr[-8000:]},
            "apollo": apollo,
            "result_format": "JSON metrics/detections; uploaded input was image data",
        }
        write_json(result_path, result)
        return 0
    except Exception as exc:
        result = {
            "job_id": job.get("job_id", job_dir.name),
            "status": "failed",
            "evaluation_mode": "yolov5_image_inference",
            "error": str(exc),
            "started_at": started,
            "completed_at": now_iso(),
            "manifest": manifest,
            "task_config": task_config,
        }
        write_json(result_path, result)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
