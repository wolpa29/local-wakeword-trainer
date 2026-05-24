import argparse
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


OUTPUT_DIR = Path("mobile_uploads/downloaded_backgrounds")
DOWNLOAD_DIR = Path("artifacts/downloads/background_sources")

SAMPLE_RATE = "16000"
CHANNELS = "1"
SAMPLE_FORMAT = "s16"

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
SPEECH_COMMANDS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found. Install it first with `sudo apt install ffmpeg`.")


def download_file(url: str, output_file: Path) -> Path:
    if output_file.exists():
        return output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with output_file.open("wb") as file, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=output_file.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    progress.update(len(chunk))

    return output_file


def convert_to_background(input_file: Path, output_name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / output_name

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
        CHANNELS,
        "-ar",
        SAMPLE_RATE,
        "-sample_fmt",
        SAMPLE_FORMAT,
        str(output_file),
    ]

    subprocess.run(command, check=True)


def add_esc50() -> int:
    archive = download_file(ESC50_URL, DOWNLOAD_DIR / "esc50.zip")
    created = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(temp_dir)

        audio_dir = Path(temp_dir) / "ESC-50-master" / "audio"
        for index, wav_file in enumerate(sorted(audio_dir.glob("*.wav")), start=1):
            convert_to_background(wav_file, f"esc50_{index:04d}.wav")
            created += 1

    return created


def add_musan() -> int:
    api = HfApi()
    files = [
        file
        for file in api.list_repo_files("FluidInference/musan", repo_type="dataset")
        if file.startswith("noise/free-sound/") and file.endswith(".wav")
    ]
    created = 0
    for index, repo_file in enumerate(sorted(files), start=1):
        local_file = hf_hub_download(
            repo_id="FluidInference/musan",
            repo_type="dataset",
            filename=repo_file,
            local_dir=DOWNLOAD_DIR / "musan",
        )
        convert_to_background(Path(local_file), f"musan_{index:04d}.wav")
        created += 1

    return created


def add_speech_commands() -> int:
    archive = download_file(SPEECH_COMMANDS_URL, DOWNLOAD_DIR / "speech_commands_v0.02.tar.gz")
    created = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(archive) as tar_file:
            members = [
                member
                for member in tar_file.getmembers()
                if member.isfile() and "_background_noise_/" in member.name and member.name.endswith(".wav")
            ]
            for index, member in enumerate(sorted(members, key=lambda item: item.name), start=1):
                extracted = tar_file.extractfile(member)
                if extracted is None:
                    continue

                input_file = Path(temp_dir) / Path(member.name).name
                input_file.write_bytes(extracted.read())
                convert_to_background(input_file, f"speech_commands_{index:03d}.wav")
                created += 1

    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public background clips for wake word training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_ffmpeg()

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    print(f"Output: {OUTPUT_DIR}")
    print("Preparing all available background clips.")
    print()

    total = 0
    created = add_musan()
    total += created
    print(f"[done] MUSAN: {created} added")

    created = add_esc50()
    total += created
    print(f"[done] ESC-50: {created} added")

    created = add_speech_commands()
    total += created
    print(f"[done] Speech Commands background: {created} added")

    print()
    print(f"Done. Added {total} background clips.")


if __name__ == "__main__":
    main()
