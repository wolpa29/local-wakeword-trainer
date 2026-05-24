import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
OPENWAKEWORD_PACKAGE = "openwakeword==0.6.0"

PIPELINE_SCRIPTS = [
    PROJECT_ROOT / "scripts" / "convert_and_cut_mobile_rec.py",
    PROJECT_ROOT / "scripts" / "augment_recordings.py",
    PROJECT_ROOT / "scripts" / "train_openwakeword.py",
]


def venv_python_path() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"

    return VENV_DIR / "bin" / "python"


def run(command: list[str], description: str) -> None:
    print(flush=True)
    print(f"==> {description}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def ensure_venv() -> Path:
    python_path = venv_python_path()

    if python_path.exists():
        print(f"Found virtual environment: {VENV_DIR}", flush=True)
        return python_path

    print(f"Creating virtual environment: {VENV_DIR}", flush=True)
    venv.create(VENV_DIR, with_pip=True)

    if not python_path.exists():
        raise RuntimeError(f"Could not find Python inside the virtual environment: {python_path}")

    return python_path


def install_requirements(python_path: Path) -> None:
    if not REQUIREMENTS_FILE.exists():
        print("No requirements.txt found, skipping install.", flush=True)
        return

    run(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--quiet",
            "-r",
            str(REQUIREMENTS_FILE),
        ],
        "Installing requirements",
    )
    run(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            OPENWAKEWORD_PACKAGE,
        ],
        "Installing openWakeWord without legacy TFLite runtime dependency",
    )


def build_train_command(python_path: Path) -> list[str]:
    return [str(python_path), str(PROJECT_ROOT / "scripts" / "train_openwakeword.py")]


def build_convert_command(python_path: Path) -> list[str]:
    return [str(python_path), str(PROJECT_ROOT / "scripts" / "convert_and_cut_mobile_rec.py")]


def run_pipeline(python_path: Path) -> None:
    for script in PIPELINE_SCRIPTS:
        if not script.exists():
            raise FileNotFoundError(f"Script not found: {script}")

        if script.name == "convert_and_cut_mobile_rec.py":
            command = build_convert_command(python_path)
        elif script.name == "train_openwakeword.py":
            command = build_train_command(python_path)
        else:
            command = [str(python_path), str(script)]

        run(command, f"Running {script.relative_to(PROJECT_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Creates .venv, installs requirements, and runs the audio pipeline."
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only create .venv and install requirements.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Do not install requirements.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    python_path = ensure_venv()

    if not args.skip_install:
        install_requirements(python_path)

    if args.setup_only:
        print(flush=True)
        print("Setup done.", flush=True)
        return

    run_pipeline(python_path)

    print(flush=True)
    print("Pipeline done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(flush=True)
        print(f"Error: command failed with exit code {error.returncode}.", flush=True)
        sys.exit(error.returncode)
    except Exception as error:
        print(flush=True)
        print(f"Error: {error}", flush=True)
        sys.exit(1)
