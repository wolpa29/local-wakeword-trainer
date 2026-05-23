import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample


# ===== Einstellungen =====

INPUT_DIR = Path("data/raw/customword")
OUTPUT_DIR = Path("data/augmented/customword")

AUGMENTATIONS_PER_FILE = 10

TARGET_SAMPLE_RATE = 16000

# Nicht zu aggressiv machen, damit das wort natürlich bleibt.
VOLUME_RANGE = (0.75, 1.35)
SPEED_RANGE = (0.92, 1.08)
NOISE_LEVEL_RANGE = (0.001, 0.008)

SUPPORTED_EXTENSIONS = {".wav"}


# ===== Hilfsfunktionen =====

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    return audio, sample_rate


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio

    return audio / peak * 0.9


def change_volume(audio: np.ndarray) -> np.ndarray:
    factor = random.uniform(*VOLUME_RANGE)
    return audio * factor


def add_noise(audio: np.ndarray) -> np.ndarray:
    noise_level = random.uniform(*NOISE_LEVEL_RANGE)
    noise = np.random.normal(0, noise_level, size=audio.shape).astype(np.float32)
    return audio + noise


def change_speed(audio: np.ndarray) -> np.ndarray:
    speed = random.uniform(*SPEED_RANGE)

    new_length = int(len(audio) / speed)

    if new_length <= 0:
        return audio

    changed = resample(audio, new_length).astype(np.float32)

    return changed


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    if len(audio) > target_length:
        start = (len(audio) - target_length) // 2
        return audio[start:start + target_length]

    if len(audio) < target_length:
        missing = target_length - len(audio)
        left = missing // 2
        right = missing - left
        return np.pad(audio, (left, right), mode="constant")

    return audio


def apply_random_augmentation(audio: np.ndarray) -> np.ndarray:
    augmented = audio.copy()

    if random.random() < 0.9:
        augmented = change_volume(augmented)

    if random.random() < 0.8:
        augmented = add_noise(augmented)

    if random.random() < 0.6:
        original_length = len(augmented)
        augmented = change_speed(augmented)
        augmented = pad_or_trim(augmented, original_length)

    augmented = normalize_audio(augmented)

    return augmented


def find_wav_files() -> list[Path]:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input-Ordner nicht gefunden: {INPUT_DIR}")

    return sorted(
        file
        for file in INPUT_DIR.rglob("*.wav")
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wav_files = find_wav_files()

    if not wav_files:
        print(f"Keine WAV-Dateien in {INPUT_DIR} gefunden.")
        return

    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Dateien: {len(wav_files)}")
    print(f"Varianten pro Datei: {AUGMENTATIONS_PER_FILE}")
    print()

    created = 0

    for wav_file in wav_files:
        audio, sample_rate = read_wav(wav_file)

        if sample_rate != TARGET_SAMPLE_RATE:
            print(f"Warnung: {wav_file.name} hat {sample_rate} Hz statt {TARGET_SAMPLE_RATE} Hz.")

        audio = normalize_audio(audio)

        # Bewahre Unterordnerstruktur: data/raw/customword/<sub>/file.wav -> data/augmented/customword/<sub>/file_aug_001.wav
        relative = wav_file.relative_to(INPUT_DIR)
        out_subdir = OUTPUT_DIR / relative.parent
        out_subdir.mkdir(parents=True, exist_ok=True)

        for index in range(1, AUGMENTATIONS_PER_FILE + 1):
            augmented = apply_random_augmentation(audio)

            output_name = f"{wav_file.stem}_aug_{index:03d}.wav"
            output_path = out_subdir / output_name

            write_wav(output_path, augmented, sample_rate)
            created += 1

        print(f"{wav_file.name} -> {AUGMENTATIONS_PER_FILE} Varianten")

    print()
    print(f"Fertig. Erstellt: {created} Dateien.")


if __name__ == "__main__":
    main()