import copy
import contextlib
import io
import json
import logging
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message="Specified provider 'CUDAExecutionProvider'.*", category=UserWarning)
warnings.filterwarnings("ignore", message="Transforms now expect an `output_type` argument.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="Warning: input samples dtype is np.float64.*", category=UserWarning)
warnings.filterwarnings("ignore", message="`isinstance\\(treespec, LeafSpec\\)` is deprecated.*", category=FutureWarning)

import numpy as np
import soundfile as sf
import torch
import torchaudio
from numpy.lib.format import open_memmap
from tqdm import tqdm

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from openwakeword.model import Model as WakewordModel
    from openwakeword.data import augment_clips, mmap_batch_generator, trim_mmap
    from openwakeword.train import Model, convert_onnx_to_tflite
    from openwakeword.utils import AudioFeatures, download_models


# ===== Settings =====

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw" / "customword"
AUGMENTED_DIR = DATA_ROOT / "augmented" / "customword"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FEATURE_DIR = ARTIFACT_DIR / "features"
MODEL_DIR = PROJECT_ROOT / "models"

# Name used for the exported files: models/homie.onnx and models/homie.tflite.
MODEL_NAME = "homie"
TARGET_PHRASE = "hey homie"

# All audio is converted to 16 kHz mono because openWakeWord expects that.
SAMPLE_RATE = 16000
# Training examples are padded to at least this length. 32000 samples = 2 seconds.
MIN_TOTAL_LENGTH = 32000
# 0.85 means 85% of positive/negative clips are used for training, 15% for validation.
TRAIN_SPLIT = 0.85


# How many augmented copies are made while building openWakeWord feature files.
AUGMENTATION_ROUNDS = 8
# How many clips are processed at once during feature building. Lower this if you run out of VRAM/RAM.
FEATURE_BATCH_SIZE = 64
# More steps can improve the model, but also makes training take longer.
# 20000+ steps recommended for good quality models
TRAIN_STEPS = 25000
# Rebuild feature files on every normal run so changed audio files are used.
OVERWRITE_FEATURES = True
# TFLite export only works with the legacy Python 3.10 stack. ONNX is the main output on Python 3.12.
MAKE_TFLITE = False
# Keep the validation report enabled by default.
RUN_EVAL = True
# Training batch size. Lower this if the GPU runs out of memory.
# Larger batches (512+) train faster and often better with sufficient RAM/GPU
BATCH_SIZE = 512
# Size of the small neural net layer
# 256 is recommended for better model capacity (default in openwakeword)
LAYER_SIZE = 256
# Learning rate for training. Higher = faster but riskier, Lower = slower but more stable.
LEARNING_RATE = 0.0001
# openWakeWord's simple dense model type
MODEL_TYPE = "dnn"
FEATURE_INFERENCE_FRAMEWORK = "onnx"
THRESHOLD_REPORT_VALUES = [round(value, 2) for value in np.arange(0.05, 1.0, 0.05)]


@dataclass
class AudioInfo:
    num_frames: int
    sample_rate: int


@dataclass
class TrainingFiles:
    positive: list[Path]
    negative: list[Path]
    background: list[Path]
    rir: list[Path]


@dataclass
class FeatureFiles:
    positive_train: Path
    positive_val: Path
    negative_train: Path
    negative_val: Path


def can_convert_tflite() -> bool:
    return sys.version_info < (3, 11)


def install_torchaudio_info_compat() -> None:
    if hasattr(torchaudio, "info"):
        return

    def info(path: str) -> AudioInfo:
        metadata = sf.info(path)
        return AudioInfo(num_frames=metadata.frames, sample_rate=metadata.samplerate)

    torchaudio.info = info


def wav_files(*parts: str) -> list[Path]:
    """Return WAV files from both raw and locally augmented folders."""
    directories = [base.joinpath(*parts) for base in (RAW_DIR, AUGMENTED_DIR)]
    return sorted(
        path
        for directory in directories
        if directory.exists()
        for path in directory.rglob("*.wav")
        if path.is_file()
    )


def split_files(files: list[Path], train_split: float) -> tuple[list[Path], list[Path]]:
    if len(files) < 2:
        return files, files

    rng = np.random.default_rng(42)
    shuffled = list(files)
    rng.shuffle(shuffled)

    split_at = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_split)))
    return shuffled[:split_at], shuffled[split_at:]


def read_duration_samples(path: Path) -> int:
    audio_info = sf.info(path)

    if audio_info.samplerate != SAMPLE_RATE:
        raise ValueError(f"{path} has {audio_info.samplerate} Hz instead of {SAMPLE_RATE} Hz.")

    return int(audio_info.frames)


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

    compute_features_from_generator_onnx(
        generator,
        n_total=n_total,
        clip_duration=total_length,
        output_file=str(output_file),
        device="gpu" if torch.cuda.is_available() else "cpu",
        ncpu=max(1, (torch.get_num_threads() or 2) // 2),
    )

    return output_file


def write_feature_batch(
    feature_file: np.memmap,
    row_counter: int,
    audio_batch: np.ndarray,
    features_model: AudioFeatures,
    batch_size: int,
    ncpu: int,
    n_total: int,
) -> int:
    features = features_model.embed_clips(audio_batch, batch_size=batch_size, ncpu=ncpu)

    if row_counter + features.shape[0] > n_total:
        features = features[0:n_total - row_counter]

    feature_file[row_counter:row_counter + features.shape[0], :, :] = features
    feature_file.flush()

    return row_counter + features.shape[0]


def compute_features_from_generator_onnx(
    generator,
    n_total: int,
    clip_duration: int,
    output_file: str,
    device: str = "cpu",
    ncpu: int = 1,
) -> None:
    features_model = AudioFeatures(
        device=device,
        ncpu=ncpu,
        inference_framework=FEATURE_INFERENCE_FRAMEWORK,
    )
    n_feature_cols = features_model.get_embedding_shape(clip_duration / SAMPLE_RATE)
    output_shape = (n_total, n_feature_cols[0], n_feature_cols[1])
    feature_file = open_memmap(output_file, mode="w+", dtype=np.float32, shape=output_shape)

    row_counter = 0
    first_audio_batch = next(generator)
    batch_size = first_audio_batch.shape[0]

    if batch_size > n_total:
        raise ValueError(
            f"The value of n_total ({n_total}) is less than the batch size ({batch_size}). "
            "Increase n_total so it is at least the batch size."
        )

    progress_bar = tqdm(total=math.ceil(n_total / batch_size), desc="Computing features")

    row_counter = write_feature_batch(
        feature_file, row_counter, first_audio_batch, features_model, batch_size, ncpu, n_total
    )
    progress_bar.update(1)

    for audio_batch in generator:
        if row_counter >= n_total:
            break

        row_counter = write_feature_batch(
            feature_file, row_counter, audio_batch, features_model, batch_size, ncpu, n_total
        )
        progress_bar.update(1)

    progress_bar.close()
    trim_mmap(output_file)


def load_features(path: Path) -> np.ndarray:
    return np.load(path)


def create_label_function(value: int):
    return lambda batch: [value for _ in batch]


class FeatureBatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_generator) -> None:
        super().__init__()
        self.batch_generator = batch_generator

    def __iter__(self):
        return self.batch_generator


def train_model(
    positive_train_features: Path,
    negative_train_features: Path,
    positive_val_features: Path,
    negative_val_features: Path,
    total_length: int,
    steps: int,
) -> torch.nn.Module:
    features = AudioFeatures(device="cpu", inference_framework=FEATURE_INFERENCE_FRAMEWORK)
    input_shape = features.get_embedding_shape(total_length / SAMPLE_RATE)

    print(f"[train] Input shape: {input_shape}")
    print(f"[train] CUDA: {'yes' if torch.cuda.is_available() else 'no'}")

    batch_generator = mmap_batch_generator(
        {
            "1": str(positive_train_features),
            "0": str(negative_train_features),
        },
        batch_size=BATCH_SIZE,
        n_per_class={"1": BATCH_SIZE // 2, "0": BATCH_SIZE // 2},
        label_transform_funcs={
            "1": create_label_function(1),
            "0": create_label_function(0),
        },
    )

    workers = 0 if torch.cuda.is_available() else max(1, min(4, torch.get_num_threads() // 2))
    x_train = torch.utils.data.DataLoader(
        FeatureBatchDataset(batch_generator),
        batch_size=None,
        num_workers=workers,
    )

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
        warmup_steps=max(1, steps // 10),
        hold_steps=max(1, steps // 4),
        lr=LEARNING_RATE,
    )

    if model.best_models:
        print(f"[train] Averaging {len(model.best_models)} good checkpoints.")
        return model.average_models(models=model.best_models)

    return copy.deepcopy(model.model)


def export_models(model: torch.nn.Module, model_name: str, total_length: int, make_tflite: bool) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    features = AudioFeatures(device="cpu", inference_framework=FEATURE_INFERENCE_FRAMEWORK)
    input_shape = features.get_embedding_shape(total_length / SAMPLE_RATE)
    onnx_path = MODEL_DIR / f"{model_name}.onnx"

    print(f"[export] Saving ONNX model: {onnx_path}")
    model_to_save = copy.deepcopy(model).to("cpu")
    model_to_save.eval()
    example_input = torch.rand(input_shape)[None,]

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        torch.onnx.export(
            model_to_save,
            example_input,
            str(onnx_path),
            opset_version=18,
            dynamo=False,
        )

    print(f"[export] ONNX created: {onnx_path}")

    if not make_tflite:
        return onnx_path

    if not can_convert_tflite():
        print("[export] Skipping TFLite: the bundled conversion stack only supports Python 3.10.")
        return onnx_path

    tflite_path = MODEL_DIR / f"{model_name}.tflite"

    try:
        convert_onnx_to_tflite(str(onnx_path), str(tflite_path))
        print(f"[export] TFLite created: {tflite_path}")
    except Exception as error:
        print(f"[export] TFLite conversion failed: {error}")
        print("[export] ONNX is still ready. Install the TFLite conversion deps and run again.")

    return onnx_path


def normalize_prediction_scores(predictions, model_name: str) -> list[float]:
    if isinstance(predictions, list):
        return [
            item.get(model_name, 0.0)
            for item in predictions
            if isinstance(item, dict)
        ]

    if isinstance(predictions, dict):
        scores = predictions.get(model_name, [])

        if isinstance(scores, np.ndarray):
            return scores.astype(float).tolist()

        if isinstance(scores, (float, int)):
            return [float(scores)]

        return list(scores)

    return []


def score_clip(model: WakewordModel, model_name: str, clip_path: Path) -> float:
    predictions = model.predict_clip(str(clip_path))
    scores = normalize_prediction_scores(predictions, model_name)

    if not scores:
        return 0.0

    return float(max(scores))


def count_scores_at_or_above(scores: list[float], threshold: float) -> int:
    return sum(score >= threshold for score in scores)


def build_threshold_report(positive_scores: list[float], negative_scores: list[float]) -> list[dict]:
    positive_total = max(1, len(positive_scores))
    negative_total = max(1, len(negative_scores))
    report = []

    for threshold in THRESHOLD_REPORT_VALUES:
        positive_hits = count_scores_at_or_above(positive_scores, threshold)
        negative_hits = count_scores_at_or_above(negative_scores, threshold)

        report.append(
            {
                "threshold": threshold,
                "positive_hits": int(positive_hits),
                "positive_total": len(positive_scores),
                "recall": positive_hits / positive_total,
                "negative_triggers": int(negative_hits),
                "negative_total": len(negative_scores),
                "false_positive_rate": negative_hits / negative_total,
            }
        )

    return report


def print_threshold_summary(threshold_report: list[dict]) -> None:
    print("[eval] Threshold check:")

    for row in threshold_report:
        if row["threshold"] not in {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}:
            continue

        print(
            f"[eval]   {row['threshold']:.2f}: "
            f"recall={row['recall']:.1%}, false positives={row['false_positive_rate']:.1%}"
        )


def evaluate_exported_model(
    model_path: Path,
    model_name: str,
    positive_files: list[Path],
    negative_files: list[Path],
) -> dict:
    print()
    print(f"[eval] Loading exported model: {model_path}")

    model = WakewordModel(wakeword_models=[str(model_path)], inference_framework="onnx")

    positive_scores = [score_clip(model, model_name, path) for path in tqdm(positive_files, desc="[eval] positive")]
    negative_scores = [score_clip(model, model_name, path) for path in tqdm(negative_files, desc="[eval] negative")]

    threshold_report = build_threshold_report(positive_scores, negative_scores)
    default_threshold = 0.5
    default_row = next(row for row in threshold_report if row["threshold"] == default_threshold)

    print("[eval] Holdout report")
    print(f"[eval] Threshold:          {default_threshold:.2f}")
    print(
        f"[eval] Positive recall:    "
        f"{default_row['positive_hits']}/{len(positive_scores)} ({default_row['recall']:.1%})"
    )
    print(
        f"[eval] Negative triggers:  "
        f"{default_row['negative_triggers']}/{len(negative_scores)} ({default_row['false_positive_rate']:.1%})"
    )

    if positive_scores:
        print(f"[eval] Positive scores:   avg={np.mean(positive_scores):.3f}, max={np.max(positive_scores):.3f}")

    if negative_scores:
        print(f"[eval] Negative scores:   avg={np.mean(negative_scores):.3f}, max={np.max(negative_scores):.3f}")

    print_threshold_summary(threshold_report)

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "threshold_report": threshold_report,
        "positive": {
            "total": len(positive_scores),
            "hits": int(default_row["positive_hits"]),
            "recall": default_row["recall"],
            "average_score": float(np.mean(positive_scores)) if positive_scores else 0.0,
            "max_score": float(np.max(positive_scores)) if positive_scores else 0.0,
            "scores": [float(score) for score in positive_scores],
            "files": [str(path.relative_to(PROJECT_ROOT)) for path in positive_files],
        },
        "negative": {
            "total": len(negative_scores),
            "triggers": int(default_row["negative_triggers"]),
            "false_positive_rate": default_row["false_positive_rate"],
            "average_score": float(np.mean(negative_scores)) if negative_scores else 0.0,
            "max_score": float(np.max(negative_scores)) if negative_scores else 0.0,
            "scores": [float(score) for score in negative_scores],
            "files": [str(path.relative_to(PROJECT_ROOT)) for path in negative_files],
        },
    }


def save_eval_result(model_path: Path, eval_result: dict) -> Path:
    eval_path = model_path.with_suffix(".eval.json")
    eval_path.write_text(json.dumps(eval_result, indent=2), encoding="utf-8")
    print(f"[eval] Saved validation result: {eval_path}")
    return eval_path


def collect_training_files() -> TrainingFiles:
    positive_files = wav_files("positive")
    negative_files = wav_files("negative")
    background_files = wav_files("background")
    rir_files = wav_files("rir")

    if background_files:
        negative_files = sorted(set(negative_files + background_files))

    return TrainingFiles(
        positive=positive_files,
        negative=negative_files,
        background=background_files,
        rir=rir_files,
    )


def check_training_files(files: TrainingFiles) -> None:
    if len(files.positive) < 20:
        raise RuntimeError(
            f"Not enough positive WAVs ({len(files.positive)}). "
            "Record at least 20-50 real wake word clips. More is better."
        )

    if len(files.negative) < 20:
        raise RuntimeError(
            f"Not enough negative/background WAVs ({len(files.negative)}). "
            "Record normal speech, TV/music, room noise, and phrases that are not the wake word."
        )


def print_training_file_summary(files: TrainingFiles) -> None:
    print(f"[data] Positive WAVs:   {len(files.positive)}")
    print(f"[data] Negative WAVs:   {len(files.negative)} (includes background if available)")
    print(f"[data] Background WAVs: {len(files.background)} (used for augmentation)")
    print(f"[data] RIR WAVs:        {len(files.rir)} (used for reverb augmentation)")


def build_all_feature_files(files: TrainingFiles, total_length: int, overwrite: bool) -> tuple[FeatureFiles, list[Path], list[Path]]:
    positive_train, positive_val = split_files(files.positive, TRAIN_SPLIT)
    negative_train, negative_val = split_files(files.negative, TRAIN_SPLIT)

    feature_files = FeatureFiles(
        positive_train=build_feature_file(
            "positive_train", positive_train, total_length, files.background, files.rir, overwrite
        ),
        positive_val=build_feature_file(
            "positive_val", positive_val, total_length, files.background, files.rir, overwrite
        ),
        negative_train=build_feature_file(
            "negative_train", negative_train, total_length, files.background, files.rir, overwrite
        ),
        negative_val=build_feature_file(
            "negative_val", negative_val, total_length, files.background, files.rir, overwrite
        ),
    )

    return feature_files, positive_val, negative_val


def main() -> None:
    install_torchaudio_info_compat()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[start] Wake phrase: {TARGET_PHRASE}")
    print(f"[start] Model name:  {MODEL_NAME}")

    download_models()

    training_files = collect_training_files()
    check_training_files(training_files)
    print_training_file_summary(training_files)

    total_length = determine_total_length(training_files.positive)
    print(f"[data] Training clip length: {total_length / SAMPLE_RATE:.2f}s")

    feature_files, positive_val, negative_val = build_all_feature_files(
        training_files,
        total_length,
        OVERWRITE_FEATURES,
    )

    print(f"[data] Feature files ready in {FEATURE_DIR}")
    print(f"[train] Using {TRAIN_STEPS} steps, batch size {BATCH_SIZE}, layer size {LAYER_SIZE}")
    print("[train] Starting training... (progress will be shown by openWakeWord)")

    trained_model = train_model(
        feature_files.positive_train,
        feature_files.negative_train,
        feature_files.positive_val,
        feature_files.negative_val,
        total_length,
        TRAIN_STEPS,
    )

    exported_model = export_models(trained_model, MODEL_NAME, total_length, make_tflite=MAKE_TFLITE)

    if RUN_EVAL:
        eval_result = evaluate_exported_model(
            exported_model,
            MODEL_NAME,
            positive_val,
            negative_val,
        )
        save_eval_result(exported_model, eval_result)

    print()
    print("[done] Done.")


if __name__ == "__main__":
    main()
