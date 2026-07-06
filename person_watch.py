import argparse
import os
from pathlib import Path
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))
(PROJECT_DIR / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T


WINDOW_NAME = "VisionTact Person Watch"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open a camera window, detect people, and show a notch warning when nobody is seen."
    )
    parser.add_argument(
        "--dfine-root",
        default=str(PROJECT_DIR / "external" / "D-FINE"),
        help="Path to a local clone of https://github.com/Peterande/D-FINE.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_DIR / "external" / "D-FINE" / "configs" / "dfine" / "dfine_hgnetv2_n_coco.yml"),
        help="D-FINE-N config path.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_DIR / "weights" / "dfine_n_coco.pth"),
        help="D-FINE-N checkpoint path.",
    )
    parser.add_argument("--source", default="0", help="Camera index or video path. Use 0 for default camera.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="D-FINE inference image size.")
    parser.add_argument(
        "--person-labels",
        default="0",
        help="Comma-separated label IDs treated as person. D-FINE COCO uses 0 for person.",
    )
    parser.add_argument(
        "--missing-seconds",
        type=float,
        default=300.0,
        help="Seconds without a person before the warning drops down.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for D-FINE, e.g. cpu, mps, cuda:0.",
    )
    return parser.parse_args()


def parse_label_ids(value):
    try:
        return {int(part.strip()) for part in value.split(",") if part.strip()}
    except ValueError as exc:
        raise ValueError("--person-labels must contain comma-separated integers.") from exc


def normalize_source(source):
    return int(source) if source.isdigit() else source


class DfineDeployModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


class DfinePersonDetector:
    def __init__(self, dfine_root, config_path, checkpoint_path, device, image_size, conf, person_labels):
        self.dfine_root = Path(dfine_root).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = torch.device(device)
        self.image_size = image_size
        self.conf = conf
        self.person_labels = person_labels
        self.transforms = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

        self._validate_paths()
        if str(self.dfine_root) not in sys.path:
            sys.path.insert(0, str(self.dfine_root))

        from src.core import YAMLConfig

        cfg = YAMLConfig(str(self.config_path), resume=str(self.checkpoint_path))
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        cfg.model.load_state_dict(state)

        self.model = DfineDeployModel(cfg).to(self.device).eval()

    def _validate_paths(self):
        missing = []
        if not self.dfine_root.exists():
            missing.append(f"D-FINE repo: {self.dfine_root}")
        if not self.config_path.exists():
            missing.append(f"D-FINE config: {self.config_path}")
        if not self.checkpoint_path.exists():
            missing.append(f"D-FINE checkpoint: {self.checkpoint_path}")

        if missing:
            missing_text = "\n".join(f"- {item}" for item in missing)
            raise FileNotFoundError(
                "Missing D-FINE deployment files:\n"
                f"{missing_text}\n\n"
                "Setup:\n"
                "  mkdir -p external weights\n"
                "  git clone https://github.com/Peterande/D-FINE.git external/D-FINE\n"
                "  .venv/bin/pip install -r external/D-FINE/requirements.txt\n"
                "  curl -L -o weights/dfine_n_coco.pth "
                "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth"
            )

    @torch.no_grad()
    def detect_people(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        width, height = image.size
        orig_size = torch.tensor([[width, height]], device=self.device)
        image_tensor = self.transforms(image).unsqueeze(0).to(self.device)

        output = self.model(image_tensor, orig_size)
        if isinstance(output, dict):
            labels = output["pred_labels"]
            boxes = output["pred_boxes"]
            scores = output["pred_scores"]
        else:
            labels, boxes, scores = output

        labels = labels[0].detach().cpu().tolist()
        boxes = boxes[0].detach().cpu().tolist()
        scores = scores[0].detach().cpu().tolist()

        people = []
        for label, box, score in zip(labels, boxes, scores):
            if int(label) in self.person_labels and float(score) >= self.conf:
                people.append({"box": box, "score": float(score)})
        return people


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
        x1, y1, x2, y2 = [int(value) for value in box["box"]]
        conf = float(box["score"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 230, 90), 2)
        label = f"person {conf:.2f}"
        cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + 130, y1), (70, 230, 90), -1)
        cv2.putText(frame, label, (x1 + 7, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    source = normalize_source(args.source)
    detector = DfinePersonDetector(
        dfine_root=args.dfine_root,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        image_size=args.imgsz,
        conf=args.conf,
        person_labels=parse_label_ids(args.person_labels),
    )

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

        person_boxes = detector.detect_people(frame)
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
