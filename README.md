# local-wakeword-trainer

Small local pipeline for training an openWakeWord wake word model from phone recordings.

Main output:

```text
models/homie.onnx
models/homie.eval.json
```

`homie.onnx` is the model. `homie.eval.json` contains the validation scores.

## Folders

Put your own recordings here:

```text
mobile_uploads/customword/positive/      wake word clips
mobile_uploads/customword/negative/      not-the-wake-word clips
mobile_uploads/customword/background/    your own room noise clips
```

Optional public background clips are stored separately:

```text
mobile_uploads/downloaded_backgrounds/
```

Generated files are written to:

```text
data/
artifacts/
models/
```

## Install System Tools

Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg python3-venv
```

## Download Public Backgrounds

This is optional, but useful for reducing false detections.

```bash
python3 download_backgrounds.py
```

This downloads and prepares public background audio from MUSAN, ESC-50, and Google Speech Commands.

## Train

```bash
python3 main.py
```

This will:

```text
1. create/use .venv
2. install Python packages
3. convert audio to 16 kHz mono WAV
4. create simple augmented clips
5. build openWakeWord features
6. train the model
7. export ONNX
8. save validation results
```

Setup only:

```bash
python3 main.py --setup-only
```

Skip package install:

```bash
python3 main.py --skip-install
```

## Settings

Change training settings near the top of:

```text
scripts/train_openwakeword.py
```

Important settings:

```text
MODEL_NAME
TARGET_PHRASE
TRAIN_STEPS
OVERWRITE_FEATURES
RUN_EVAL
```

Change downloaded background usage near the top of:

```text
scripts/convert_and_cut_mobile_rec.py
```

Important settings:

```text
DEFAULT_DOWNLOADED_BACKGROUND_COUNT
DEFAULT_DOWNLOADED_BACKGROUND_SEED
```

Use `-1` for `DEFAULT_DOWNLOADED_BACKGROUND_COUNT` to use all downloaded backgrounds.

## Notes

Python 3.12 is supported with ONNX export. TFLite export is disabled by default because the old openWakeWord TFLite stack works best on Python 3.10.

Do not commit private recordings, generated data, or trained models unless you really want to publish them.
