import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    class tqdm:
        def __init__(self, total=None, desc=None):
            self.total = total
            self.n = 0
        def update(self, n=1):
            pass
        def close(self):
            pass


# ===== Settings =====

INPUT_DIR = Path("mobile_uploads/customword")
DOWNLOADED_BACKGROUND_DIR = Path("mobile_uploads/downloaded_backgrounds")
OUTPUT_DIR = Path("data/raw/customword")

DEFAULT_DOWNLOADED_BACKGROUND_COUNT = None  # None means use all available backgrounds
DEFAULT_DOWNLOADED_BACKGROUND_SEED = 42

TARGET_SAMPLE_RATE = "16000"
TARGET_CHANNELS = "1"
TARGET_SAMPLE_FORMAT = "s16"

TRIM_SILENCE = True
NO_TRIM_FOLDERS = {"background", "rir"}

# Closer to 0 means stronger silence trimming.
# -35dB is gentle, -30dB is stronger, -25dB is pretty aggressive.
SILENCE_THRESHOLD = "-35dB"

# How long silence has to be before ffmpeg trims it.
MIN_SILENCE_DURATION = "0.15"

SUPPORTED_EXTENSIONS = {
    ".m4a",
    ".aac",
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".webm",
    ".mp4",
    ".caf",
    ".aif",
    ".aiff",
    ".wma",
}


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found. Install it first, for example with `sudo apt install ffmpeg`."
        )


def clean_filename(name: str) -> str:
    name = name.strip().lower()

    replacements = {
        "\u00e4": "ae",
        "\u00f6": "oe",
        "\u00fc": "ue",
        "\u00df": "ss",
        " ": "_",
        "-": "_",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    allowed_chars = "abcdefghijklmnopqrstuvwxyz0123456789_"
    cleaned = "".join(char for char in name if char in allowed_chars)

    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    return cleaned.strip("_") or "recording"


def find_audio_files_in(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    return sorted(
        file
        for file in directory.rglob("*")
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def choose_downloaded_background_files(files: list[Path], count: int | None, seed: int) -> list[Path]:
    if count is None:
        # Use all available files, with random selection
        rng = random.Random(seed)
        shuffled = sorted(files)
        rng.shuffle(shuffled)
        return shuffled

    if count < 0 or count >= len(files):
        return files

    rng = random.Random(seed)
    shuffled = sorted(files)
    rng.shuffle(shuffled)
    return sorted(shuffled[:count])


def find_audio_files() -> list[Path]:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    own_files = find_audio_files_in(INPUT_DIR)
    downloaded_background_files = choose_downloaded_background_files(
        find_audio_files_in(DOWNLOADED_BACKGROUND_DIR),
        DEFAULT_DOWNLOADED_BACKGROUND_COUNT,
        DEFAULT_DOWNLOADED_BACKGROUND_SEED,
    )

    return own_files + downloaded_background_files


def is_inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def should_trim_silence(input_file: Path) -> bool:
    if is_inside(input_file, DOWNLOADED_BACKGROUND_DIR):
        return False

    relative_parts = input_file.relative_to(INPUT_DIR).parts
    return TRIM_SILENCE and not any(part in NO_TRIM_FOLDERS for part in relative_parts)


def output_path_for(input_file: Path) -> Path:
    if is_inside(input_file, DOWNLOADED_BACKGROUND_DIR):
        clean_stem = clean_filename(input_file.stem)
        return OUTPUT_DIR / "background" / f"downloaded_{clean_stem}.wav"

    relative = input_file.relative_to(INPUT_DIR)
    clean_stem = clean_filename(input_file.stem)
    return OUTPUT_DIR / relative.parent / f"{clean_stem}.wav"


def build_audio_filter(trim_silence: bool) -> str:
    filters = []

    if trim_silence:
        # Gently trim silence from the beginning and end.
        filters.append(
            (
                f"silenceremove="
                f"start_periods=1:"
                f"start_duration={MIN_SILENCE_DURATION}:"
                f"start_threshold={SILENCE_THRESHOLD}:"
                f"stop_periods=1:"
                f"stop_duration={MIN_SILENCE_DURATION}:"
                f"stop_threshold={SILENCE_THRESHOLD}"
            )
        )

    return ",".join(filters)


def convert_to_wav(input_file: Path, output_file: Path, trim_silence: bool) -> None:
    # Simple progress indicator via tqdm wrapper
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_file),
        "-vn",
        "-ac",
        TARGET_CHANNELS,
        "-ar",
        TARGET_SAMPLE_RATE,
        "-sample_fmt",
        TARGET_SAMPLE_FORMAT,
    ]

    audio_filter = build_audio_filter(trim_silence)

    if audio_filter:
        command.extend(["-af", audio_filter])

    command.append(str(output_file))

    subprocess.run(command, check=True)


def main() -> None:
    check_ffmpeg()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = find_audio_files()

    if not audio_files:
        print(f"No audio files found in {INPUT_DIR}.")
        return

    # Count files by type for summary
    own_count = sum(1 for f in audio_files if not is_inside(f, DOWNLOADED_BACKGROUND_DIR))
    downloaded_count = len(audio_files) - own_count

    print(f"Input:              {INPUT_DIR}")
    print(f"Extra background:   {DOWNLOADED_BACKGROUND_DIR}")
    print(f"Downloaded count:   {DEFAULT_DOWNLOADED_BACKGROUND_COUNT} ({downloaded_count} used)")
    print(f"Downloaded seed:    {DEFAULT_DOWNLOADED_BACKGROUND_SEED}")
    print(f"Output:             {OUTPUT_DIR}")
    print(f"Total files:        {len(audio_files)}")
    print(f"Trim silence:       {TRIM_SILENCE} (except {', '.join(sorted(NO_TRIM_FOLDERS))})")
    print()

    pbar = tqdm(total=len(audio_files), desc="Converting audio", unit="file")
    for input_file in audio_files:
        output_file = output_path_for(input_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        trim_silence = should_trim_silence(input_file)

        convert_to_wav(input_file, output_file, trim_silence)
        pbar.update(1)

    pbar.close()
    print()
    print(f"Done. Processed {own_count} own + {downloaded_count} downloaded files.")


if __name__ == "__main__":
    main()
