import argparse
import copy
import logging
import math
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch
from tqdm import tqdm

from openwakeword.data import augment_clips, mmap_batch_generator
from openwakeword.train import Model, convert_onnx_to_tflite
from openwakeword.utils import AudioFeatures, compute_features_from_generator, download_models


# ===== Settings =====

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw" / "customword"
AUGMENTED_DIR = DATA_ROOT / "augmented" / "customword"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FEATURE_DIR = ARTIFACT_DIR / "features"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_NAME = "homie"
TARGET_PHRASE = "hey homie"

SAMPLE_RATE = 16000
MIN_TOTAL_LENGTH = 32000
TRAIN_SPLIT = 0.85

AUGMENTATION_ROUNDS = 8
FEATURE_BATCH_SIZE = 64

TRAIN_STEPS = 6000
BATCH_SIZE = 256
LAYER_SIZE = 128
MODEL_TYPE = "dnn"


def wav_files(*parts: str) -> list[Path]:
    files: list[Path] = []

    for base in (RAW_DIR, AUGMENTED_DIR):
        directory = base.joinpath(*parts)
        if directory.exists():
            files.extend(directory.rglob("*.wav"))

    return sorted(path for path in files if path.is_file())


def split_files(files: list[Path], train_split: float) -> tuple[list[Path], list[Path]]:
    if len(files) < 2:
        return files, files

    rng = np.random.default_rng(42)
    shuffled = list(files)
    rng.shuffle(shuffled)

    split_at = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_split)))
    return shuffled[:split_at], shuffled[split_at:]


def read_duration_samples(path: Path) -> int:
    sample_rate, audio = scipy.io.wavfile.read(path)

    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path} hat {sample_rate} Hz statt {SAMPLE_RATE} Hz.")

    return int(audio.shape[0])


def determine_total_length(positive_files: list[Path]) -> int:
    durations = [read_duration_samples(path) for path in positive_files]
    median_duration = int(np.median(durations))
    total_length = int(round(median_duration / 1000) * 1000) + 12000

    if total_length < MIN_TOTAL_LENGTH or abs(total_length - MIN_TOTAL_LENGTH) <= 4000:
        return MIN_TOTAL_LENGTH

    return total_length


def build_feature_file(
    name: str,
    files: list[Path],
    total_length: int,
    background_files: list[Path],
    rir_files: list[Path],
    overwrite: bool,
) -> Path:
    output_file = FEATURE_DIR / f"{name}.npy"

    if output_file.exists() and not overwrite:
        print(f"[features] Using existing file: {output_file}")
        return output_file

    repeated_files = [str(path) for path in files] * AUGMENTATION_ROUNDS
    n_total = len(repeated_files)

    if n_total < FEATURE_BATCH_SIZE:
        repeat_factor = math.ceil(FEATURE_BATCH_SIZE / max(1, n_total))
        repeated_files *= repeat_factor
        n_total = len(repeated_files)

    print(f"[features] {name}: {len(files)} clips x {AUGMENTATION_ROUNDS} rounds -> {n_total} examples")

    generator = augment_clips(
        repeated_files,
        total_length=total_length,
        batch_size=min(FEATURE_BATCH_SIZE, n_total),
        background_clip_paths=[str(path) for path in background_files],
        RIR_paths=[str(path) for path in rir_files],
    )

    compute_features_from_generator(
        generator,
        n_total=n_total,
        clip_duration=total_length,
        output_file=str(output_file),
        device="gpu" if torch.cuda.is_available() else "cpu",
        ncpu=max(1, (torch.get_num_threads() or 2) // 2),
    )

    return output_file


def load_features(path: Path) -> np.ndarray:
    return np.load(path)


def train_model(
    positive_train_features: Path,
    negative_train_features: Path,
    positive_val_features: Path,
    negative_val_features: Path,
    total_length: int,
    steps: int,
) -> torch.nn.Module:
    features = AudioFeatures(device="cpu")
    input_shape = features.get_embedding_shape(total_length / SAMPLE_RATE)

    print(f"[train] Input shape: {input_shape}")
    print(f"[train] CUDA: {'yes' if torch.cuda.is_available() else 'no'}")

    def label(value: int):
        return lambda x: [value for _ in x]

    batch_generator = mmap_batch_generator(
        {
            "1": str(positive_train_features),
            "0": str(negative_train_features),
        },
        batch_size=BATCH_SIZE,
        n_per_class={"1": BATCH_SIZE // 2, "0": BATCH_SIZE // 2},
        label_transform_funcs={"1": label(1), "0": label(0)},
    )

    class IterDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            return batch_generator

    workers = 0 if torch.cuda.is_available() else max(1, min(4, torch.get_num_threads() // 2))
    x_train = torch.utils.data.DataLoader(IterDataset(), batch_size=None, num_workers=workers)

    x_val_pos = load_features(positive_val_features)
    x_val_neg = load_features(negative_val_features)
    y_val = np.hstack((np.ones(x_val_pos.shape[0]), np.zeros(x_val_neg.shape[0]))).astype(np.float32)
    x_val = np.vstack((x_val_pos, x_val_neg)).astype(np.float32)

    x_val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=len(y_val),
    )

    model = Model(
        n_classes=1,
        input_shape=input_shape,
        model_type=MODEL_TYPE,
        layer_dim=LAYER_SIZE,
        seconds_per_example=1280 * input_shape[0] / SAMPLE_RATE,
    )

    val_steps = np.linspace(max(10, steps // 5), steps - 1, 12).astype(np.int64)
    weights = np.linspace(1, 1000, steps).tolist()

    model.train_model(
        X=x_train,
        X_val=x_val_loader,
        false_positive_val_data=None,
        max_steps=steps,
        negative_weight_schedule=weights,
        val_steps=val_steps,
        warmup_steps=max(1, steps // 5),
        hold_steps=max(1, steps // 3),
        lr=0.0001,
    )

    if model.best_models:
        print(f"[train] Averaging {len(model.best_models)} good checkpoints.")
        return model.average_models(models=model.best_models)

    return copy.deepcopy(model.model)


def export_models(model: torch.nn.Module, model_name: str, total_length: int, make_tflite: bool) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    features = AudioFeatures(device="cpu")
    input_shape = features.get_embedding_shape(total_length / SAMPLE_RATE)
    wrapper = Model(n_classes=1, input_shape=input_shape, model_type=MODEL_TYPE, layer_dim=LAYER_SIZE)

    wrapper.export_model(model=model, model_name=model_name, output_dir=str(MODEL_DIR))
    onnx_path = MODEL_DIR / f"{model_name}.onnx"

    print(f"[export] ONNX created: {onnx_path}")

    if not make_tflite:
        return

    tflite_path = MODEL_DIR / f"{model_name}.tflite"

    try:
        convert_onnx_to_tflite(str(onnx_path), str(tflite_path))
        print(f"[export] TFLite created: {tflite_path}")
    except Exception as error:
        print(f"[export] TFLite conversion failed: {error}")
        print("[export] ONNX is still ready. Install the TFLite conversion deps and run again.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom openWakeWord ONNX/TFLite model.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--target-phrase", default=TARGET_PHRASE)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--no-tflite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[start] Wake phrase: {args.target_phrase}")
    print(f"[start] Model name:  {args.model_name}")

    download_models()

    positive_files = wav_files("positive")
    negative_files = wav_files("negative")
    background_files = wav_files("background")
    rir_files = wav_files("rir")

    if background_files:
        negative_files = sorted(set(negative_files + background_files))

    if len(positive_files) < 20:
        raise RuntimeError(
            f"Not enough positive WAVs ({len(positive_files)}). "
            "Record at least 20-50 real wake word clips. More is better."
        )

    if len(negative_files) < 20:
        raise RuntimeError(
            f"Not enough negative/background WAVs ({len(negative_files)}). "
            "Record normal speech, TV/music, room noise, and phrases that are not the wake word."
        )

    print(f"[data] Positive WAVs:   {len(positive_files)}")
    print(f"[data] Negative WAVs:   {len(negative_files)}")
    print(f"[data] Background WAVs: {len(background_files)}")
    print(f"[data] RIR WAVs:        {len(rir_files)}")

    total_length = determine_total_length(positive_files)
    print(f"[data] Training clip length: {total_length / SAMPLE_RATE:.2f}s")

    positive_train, positive_val = split_files(positive_files, TRAIN_SPLIT)
    negative_train, negative_val = split_files(negative_files, TRAIN_SPLIT)

    positive_train_features = build_feature_file(
        "positive_train", positive_train, total_length, background_files, rir_files, args.overwrite_features
    )
    positive_val_features = build_feature_file(
        "positive_val", positive_val, total_length, background_files, rir_files, args.overwrite_features
    )
    negative_train_features = build_feature_file(
        "negative_train", negative_train, total_length, background_files, rir_files, args.overwrite_features
    )
    negative_val_features = build_feature_file(
        "negative_val", negative_val, total_length, background_files, rir_files, args.overwrite_features
    )

    trained_model = train_model(
        positive_train_features,
        negative_train_features,
        positive_val_features,
        negative_val_features,
        total_length,
        args.steps,
    )

    export_models(trained_model, args.model_name, total_length, make_tflite=not args.no_tflite)

    print()
    print("[done] Done.")


if __name__ == "__main__":
    main()
