# local-wakeword-trainer

Small pipeline for training my own openWakeWord-compatible wake word model from phone recordings.

The goal is to end up with:

```text
models/homie.onnx
models/homie.tflite
```

Those files can then be used on a Raspberry Pi with openWakeWord. This repo is not tied to Home Assistant. The plan is just: detect the wake word locally, then hand off to whatever voice assistant / LLM stack I want to run.

## Recording folders

Drop raw phone recordings here:

```text
mobile_uploads/customword/
  positive/      The wake word, for example "hey homie" or "homie"
  negative/      Speech without the wake word, similar phrases, normal commands
  background/    Room noise, TV, music, keyboard, kitchen noise, silence
  rir/           Optional room impulse response WAVs
```

Common phone formats like `.m4a`, `.mp3`, `.wav`, `.flac`, and `.ogg` are found recursively.

`background/` and `rir/` clips are converted to 16 kHz WAV, but they are not silence-trimmed or duplicated by the simple augmentation script. They are used later as room/noise material while building training features.

For a first useful run I would start around here:

```text
positive:   at least 20-50 real clips, 100+ is better
negative:   at least 50 clips, a few hundred is better
background: at least 10 longer clips from real rooms
```

Very short wake words like `homie` are easier to trigger by accident. `hey homie` should usually be more stable.

## Run the pipeline

Use Python 3.10 on the Ubuntu/RTX training machine:

```bash
python3.10 main.py
```

The pipeline does this:

```text
1. create .venv
2. install requirements
3. convert mobile_uploads/customword/* to data/raw/customword/*
4. augment data/raw/customword/* into data/augmented/customword/*
5. build openWakeWord features
6. train the model
7. export ONNX and, if possible, TFLite
```

Setup only:

```bash
python3.10 main.py --setup-only
```

Run only the training script:

```bash
.venv/bin/python scripts/train_openwakeword.py --model-name homie --target-phrase "hey homie"
```

Rebuild feature files:

```bash
.venv/bin/python scripts/train_openwakeword.py --overwrite-features
```

Only export ONNX if the TFLite conversion is being annoying:

```bash
.venv/bin/python scripts/train_openwakeword.py --no-tflite
```

## Raspberry Pi usage

The finished files should be here:

```text
models/homie.onnx
models/homie.tflite
```

In my own Python code I can load the model like this:

```python
from openwakeword.model import Model

model = Model(wakeword_models=["models/homie.tflite"])
```

The audio stream needs to be 16 kHz, mono, 16-bit PCM.

## Setup notes

`ffmpeg` is needed for converting phone recordings.

Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg python3.10 python3.10-venv
```

Windows is fine for sorting files and checking the repo, but the full training pipeline is meant for Ubuntu with Python 3.10.

## Public repo / license notes

This repo only tracks the pipeline code. Audio recordings, feature files, training artifacts, and exported models are intentionally ignored:

```text
mobile_uploads/
data/
artifacts/
models/
```

This project depends on openWakeWord. The openWakeWord code is Apache-2.0 licensed. The pretrained models shipped by openWakeWord can have different, non-commercial license terms, so check their license notes before using or publishing trained model weights commercially.

Do not publish private voice recordings, background recordings, or generated model files unless you know you have the rights to do that.
