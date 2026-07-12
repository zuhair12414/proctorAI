#!/usr/bin/env python3
"""Line-delimited JSON local vision worker for the Riya LiveKit agent.

Protocol:
  stdin  line: {"id": 1, "image_base64": "...jpeg..."}
  stdout line: {"id": 1, "result": {...}}

All logs go to stderr so stdout stays machine-readable.
"""

import base64
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from person_watch import DfinePersonDetector, parse_label_ids

PROJECT_DIR = Path(__file__).resolve().parent

DFINE_ROOT = os.getenv("DFINE_ROOT", str(PROJECT_DIR / "external" / "D-FINE"))
DFINE_CONFIG = os.getenv(
    "DFINE_CONFIG",
    str(PROJECT_DIR / "external" / "D-FINE" / "configs" / "dfine" / "dfine_hgnetv2_n_coco.yml"),
)
DFINE_CHECKPOINT = os.getenv("DFINE_CHECKPOINT", str(PROJECT_DIR / "weights" / "dfine_n_coco.pth"))
VISION_DEVICE = os.getenv("VISION_DEVICE", "cpu")
VISION_IMGSZ = int(os.getenv("VISION_IMGSZ", "640"))
VISION_CONF = float(os.getenv("VISION_CONF", "0.35"))
VISION_PERSON_LABELS = os.getenv("VISION_PERSON_LABELS", "0")
VISION_NMS_IOU = float(os.getenv("VISION_NMS_IOU", "0.5"))
PARTIAL_EDGE_THRESHOLD = float(os.getenv("PARTIAL_EDGE_THRESHOLD", "0.035"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.45"))


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def decode_jpeg_base64(image_base64: str) -> np.ndarray:
    image_bytes = base64.b64decode(image_base64)
    np_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode JPEG frame.")
    return frame


def normalize_box(box: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        clamp01(x1 / max(width, 1)),
        clamp01(y1 / max(height, 1)),
        clamp01(x2 / max(width, 1)),
        clamp01(y2 / max(height, 1)),
    ]


def box_area(box: List[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def touches_edge(norm_box: List[float], threshold: float) -> bool:
    if len(norm_box) != 4:
        return False

    x1, y1, x2, y2 = norm_box

    # Interview framing rule:
    # Ignore the bottom edge because the candidate's lower body/shoulders
    # will normally be outside the webcam frame. Only flag partial_frame
    # when the person is cut off on the left, top, or right side.
    return x1 <= threshold or y1 <= threshold or x2 >= 1.0 - threshold


def summarize_detection(people: List[Dict[str, Any]], width: int, height: int, inference_ms: float) -> Dict[str, Any]:
    person_count = len(people)

    normalized_people: List[Dict[str, Any]] = []
    for person in people:
        raw_box = [float(value) for value in person["box"]]
        norm_box = normalize_box(raw_box, width, height)
        normalized_people.append(
            {
                "bbox": norm_box,
                "score": float(person["score"]),
                "area": box_area(norm_box),
            }
        )

    normalized_people.sort(key=lambda item: (item["score"], item["area"]), reverse=True)
    primary: Optional[Dict[str, Any]] = normalized_people[0] if normalized_people else None

    if person_count == 0:
        return {
            "person_count": 0,
            "status": "no_person",
            "confidence": 0,
            "bbox": [],
            "boxes": [],
            "reason": "No person detected by local D-FINE model.",
            "inference_ms": round(inference_ms, 2),
        }

    primary_bbox = primary["bbox"]
    primary_confidence = float(primary["score"])

    if person_count > 1:
        status = "multiple_persons"
        reason = f"{person_count} people detected by local D-FINE model."
    elif primary_confidence < LOW_CONFIDENCE_THRESHOLD:
        status = "low_confidence"
        reason = f"One person detected locally, but confidence is low ({primary_confidence:.2f})."
    elif touches_edge(primary_bbox, PARTIAL_EDGE_THRESHOLD):
        status = "partial_frame"
        reason = "One person detected, but the bounding box is touching the left, top, or right frame edge."
    else:
        status = "ok"
        reason = "One person detected locally and appears inside the frame."

    return {
        "person_count": person_count,
        "status": status,
        "confidence": round(primary_confidence, 4),
        "bbox": primary_bbox,
        "boxes": [
            {
                "bbox": item["bbox"],
                "score": round(float(item["score"]), 4),
            }
            for item in normalized_people
        ],
        "reason": reason,
        "inference_ms": round(inference_ms, 2),
    }


def build_detector() -> DfinePersonDetector:
    eprint("Loading local D-FINE detector...")
    eprint(f"  root:       {DFINE_ROOT}")
    eprint(f"  config:     {DFINE_CONFIG}")
    eprint(f"  checkpoint: {DFINE_CHECKPOINT}")
    eprint(f"  device:     {VISION_DEVICE}")

    detector = DfinePersonDetector(
        dfine_root=DFINE_ROOT,
        config_path=DFINE_CONFIG,
        checkpoint_path=DFINE_CHECKPOINT,
        device=VISION_DEVICE,
        image_size=VISION_IMGSZ,
        conf=VISION_CONF,
        person_labels=parse_label_ids(VISION_PERSON_LABELS),
        nms_iou=VISION_NMS_IOU,
    )
    eprint("Local D-FINE detector ready.")
    return detector


def process_request(detector: DfinePersonDetector, request: Dict[str, Any]) -> Dict[str, Any]:
    request_id = request.get("id")
    image_base64 = request.get("image_base64")
    if not image_base64:
        raise ValueError("Missing image_base64 in request.")

    frame = decode_jpeg_base64(image_base64)
    height, width = frame.shape[:2]

    started = time.perf_counter()
    people = detector.detect_people(frame)
    inference_ms = (time.perf_counter() - started) * 1000

    return {
        "id": request_id,
        "result": summarize_detection(people, width, height, inference_ms),
    }


def main() -> None:
    detector = build_detector()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            response = process_request(detector, request)
        except Exception as exc:  # noqa: BLE001 - worker must report all errors as JSON.
            response = {
                "id": request_id,
                "error": str(exc),
            }

        print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
