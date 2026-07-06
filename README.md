# ProctorAI

Local person-detection monitor built with D-FINE-N and OpenCV.

The app opens a camera window, detects people, and shows a top-center warning notch if no person has been detected for a configurable interval. The default interval is five minutes.

## Setup

Install the D-FINE source and D-FINE-N COCO checkpoint locally:

```bash
mkdir -p external weights
git clone https://github.com/Peterande/D-FINE.git external/D-FINE
.venv/bin/pip install -r external/D-FINE/requirements.txt
curl -L -o weights/dfine_n_coco.pth https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth
```

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

## D-FINE-N defaults

The app uses:

```text
external/D-FINE/configs/dfine/dfine_hgnetv2_n_coco.yml
weights/dfine_n_coco.pth
```

D-FINE COCO labels use `0` for `person`, so the default detection filter is:

```bash
--person-labels 0
```
