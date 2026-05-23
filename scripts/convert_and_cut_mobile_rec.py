import shutil
import subprocess
from pathlib import Path


# ===== Einstellungen =====

INPUT_DIR = Path("mobile_uploads/customword")
OUTPUT_DIR = Path("data/raw/customword")

TARGET_SAMPLE_RATE = "16000"
TARGET_CHANNELS = "1"
TARGET_SAMPLE_FORMAT = "s16"

TRIM_SILENCE = True

# Je näher an 0, desto empfindlicher.
# -35dB ist vorsichtig, -30dB etwas stärker, -25dB aggressiver.
SILENCE_THRESHOLD = "-35dB"

# Wie lange Stille erkannt werden muss, bevor getrimmt wird.
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
            "ffmpeg wurde nicht gefunden. Installiere es mit: winget install Gyan.FFmpeg"
        )


def clean_filename(name: str) -> str:
    name = name.strip().lower()

    replacements = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
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
        raise FileNotFoundError(f"Input-Ordner nicht gefunden: {INPUT_DIR}")

    return sorted(
        file
        for file in INPUT_DIR.rglob("*")
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def unique_output_path(output_dir: Path, filename_stem: str) -> Path:
    # Kehrt jetzt einfach den Pfad zurück, da wir beim Konvertieren überschreiben wollen.
    return output_dir / f"{filename_stem}.wav"


def build_audio_filter() -> str:
    filters = []

    if TRIM_SILENCE:
        # Entfernt Stille am Anfang und Ende vorsichtig.
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


def convert_to_wav(input_file: Path, output_file: Path) -> None:
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

    audio_filter = build_audio_filter()

    if audio_filter:
        command.extend(["-af", audio_filter])

    command.append(str(output_file))

    subprocess.run(command, check=True)


def main() -> None:
    check_ffmpeg()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = find_audio_files()

    if not audio_files:
        print(f"Keine Audiodateien in {INPUT_DIR} gefunden.")
        return

    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Dateien: {len(audio_files)}")
    print(f"Trim silence: {TRIM_SILENCE}")
    print()

    for input_file in audio_files:
        # Behalte die Ordnerstruktur bei: mobile_uploads/customword/<sub>/file -> data/raw/customword/<sub>/file.wav
        relative = input_file.relative_to(INPUT_DIR)
        out_subdir = OUTPUT_DIR / relative.parent
        out_subdir.mkdir(parents=True, exist_ok=True)

        clean_stem = clean_filename(input_file.stem)
        output_file = unique_output_path(out_subdir, clean_stem)

        print(f"{input_file.name} -> {output_file.relative_to(OUTPUT_DIR)}")
        convert_to_wav(input_file, output_file)

    print()
    print("Fertig.")


if __name__ == "__main__":
    main()
