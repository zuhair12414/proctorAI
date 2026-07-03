import argparse
import os
from pathlib import Path
import time

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))
(PROJECT_DIR / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
from ultralytics import YOLO


PERSON_CLASS_ID = 0
WINDOW_NAME = "VisionTact Person Watch"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open a camera window, detect people, and show a notch warning when nobody is seen."
    )
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path.")
    parser.add_argument("--source", default="0", help="Camera index or video path. Use 0 for default camera.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size.")
    parser.add_argument(
        "--missing-seconds",
        type=float,
        default=300.0,
        help="Seconds without a person before the warning drops down.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional Ultralytics device, e.g. cpu, mps, 0. Leave unset for auto.",
    )
    return parser.parse_args()


def normalize_source(source):
    return int(source) if source.isdigit() else source


def draw_rounded_rect(image, top_left, bottom_right, radius, color, thickness=-1):
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    if thickness < 0:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.circle(image, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(image, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(image, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(image, (x2 - radius, y2 - radius), radius, color, thickness)
    else:
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)


def draw_text_center(image, text, center, font_scale, color, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(center[0] - size[0] / 2)
    y = int(center[1] + (size[1] - baseline) / 2)
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_notch(frame, message, active_for_seconds):
    height, width = frame.shape[:2]
    notch_w = min(560, max(330, int(width * 0.58)))
    notch_h = 78
    notch_x = (width - notch_w) // 2

    drop_seconds = 0.45
    progress = min(1.0, active_for_seconds / drop_seconds)
    ease = 1 - pow(1 - progress, 3)
    notch_y = int(-notch_h + 14 + ease * (notch_h + 10))

    overlay = frame.copy()
    shadow_y = notch_y + 7
    draw_rounded_rect(
        overlay,
        (notch_x + 8, shadow_y + 8),
        (notch_x + notch_w - 8, shadow_y + notch_h - 2),
        26,
        (10, 10, 10),
    )

    pulse = int(35 + 25 * (0.5 + 0.5 * np.sin(time.monotonic() * 5)))
    draw_rounded_rect(
        overlay,
        (notch_x, notch_y),
        (notch_x + notch_w, notch_y + notch_h),
        28,
        (24, 24, 28 + pulse),
    )
    draw_rounded_rect(
        overlay,
        (notch_x, notch_y),
        (notch_x + notch_w, notch_y + notch_h),
        28,
        (55, 55, 210),
        thickness=2,
    )

    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    draw_text_center(frame, "NO PERSON DETECTED", (width // 2, notch_y + 30), 0.72, (255, 255, 255), 2)
    draw_text_center(frame, message, (width // 2, notch_y + 56), 0.48, (215, 220, 255), 1)


def draw_hud(frame, person_count, seconds_until_warning):
    status = f"People: {person_count}"
    if person_count:
        status += " | monitoring"
        color = (80, 220, 120)
    else:
        status += f" | warning in {max(0, int(seconds_until_warning))}s"
        color = (80, 190, 255)

    cv2.rectangle(frame, (12, 12), (440, 48), (18, 18, 18), -1)
    cv2.putText(frame, status, (24, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "Press q or Esc to quit",
        (24, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def draw_person_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 230, 90), 2)
        label = f"person {conf:.2f}"
        cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + 130, y1), (70, 230, 90), -1)
        cv2.putText(frame, label, (x1 + 7, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    source = normalize_source(args.source)
    model = YOLO(args.model)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source {args.source}. Try --source 1 or check camera permissions.")

    last_person_seen_at = time.monotonic()
    warning_started_at = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        person_boxes = list(result.boxes)
        now = time.monotonic()

        if person_boxes:
            last_person_seen_at = now
            warning_started_at = None

        seconds_missing = now - last_person_seen_at
        seconds_until_warning = args.missing_seconds - seconds_missing

        draw_person_boxes(frame, person_boxes)
        draw_hud(frame, len(person_boxes), seconds_until_warning)

        if seconds_missing >= args.missing_seconds:
            if warning_started_at is None:
                warning_started_at = now
            minutes_missing = seconds_missing / 60
            draw_notch(frame, f"No person seen for {minutes_missing:.1f} minutes", now - warning_started_at)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
