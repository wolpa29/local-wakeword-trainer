import shutil
import subprocess
from pathlib import Path


# ===== Settings =====

INPUT_DIR = Path("mobile_uploads/customword")
OUTPUT_DIR = Path("data/raw/customword")

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


def find_audio_files() -> list[Path]:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    return sorted(
        file
        for file in INPUT_DIR.rglob("*")
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def unique_output_path(output_dir: Path, filename_stem: str) -> Path:
    return output_dir / f"{filename_stem}.wav"


def should_trim_silence(input_file: Path) -> bool:
    relative_parts = input_file.relative_to(INPUT_DIR).parts
    return TRIM_SILENCE and not any(part in NO_TRIM_FOLDERS for part in relative_parts)


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
    command = [
        "ffmpeg",
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

    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Files:  {len(audio_files)}")
    print(f"Trim silence: {TRIM_SILENCE} except for {', '.join(sorted(NO_TRIM_FOLDERS))}")
    print()

    for input_file in audio_files:
        # Keep the folder layout: mobile_uploads/customword/<sub>/file -> data/raw/customword/<sub>/file.wav
        relative = input_file.relative_to(INPUT_DIR)
        out_subdir = OUTPUT_DIR / relative.parent
        out_subdir.mkdir(parents=True, exist_ok=True)

        clean_stem = clean_filename(input_file.stem)
        output_file = unique_output_path(out_subdir, clean_stem)
        trim_silence = should_trim_silence(input_file)

        print(f"{input_file.name} -> {output_file.relative_to(OUTPUT_DIR)} (trim={trim_silence})")
        convert_to_wav(input_file, output_file, trim_silence)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
