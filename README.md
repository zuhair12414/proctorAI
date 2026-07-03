# ProctorAI

Local person-detection monitor built with YOLO11n and OpenCV.

The app opens a camera window, detects people, and shows a top-center warning notch if no person has been detected for a configurable interval. The default interval is five minutes.

## Run

```bash
source .venv/bin/activate
python person_watch.py --source 0
```

For a quick warning test:

```bash
python person_watch.py --source 0 --missing-seconds 10
```

If `source 0` points to OBS or another virtual camera, try:

```bash
python person_watch.py --source 1
```

Press `q` or `Esc` to quit.
