import argparse
import subprocess
import sys
import venv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "download_background_clips.py"


def venv_python_path() -> Path:
    return VENV_DIR / "Scripts" / "python.exe" if sys.platform == "win32" else VENV_DIR / "bin" / "python"


def run(command: list[str], description: str) -> None:
    print()
    print(f"==> {description}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def ensure_venv() -> Path:
    python_path = venv_python_path()

    if python_path.exists():
        return python_path

    print(f"Creating virtual environment: {VENV_DIR}", flush=True)
    venv.create(VENV_DIR, with_pip=True)
    return python_path


def install_requirements(python_path: Path) -> None:
    run(
        [str(python_path), "-m", "pip", "install", "--quiet", "-r", str(REQUIREMENTS_FILE)],
        "Installing requirements",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public background clips for training.")
    parser.add_argument("--skip-install", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_path = ensure_venv()

    if not args.skip_install:
        install_requirements(python_path)

    run(
        [str(python_path), str(DOWNLOAD_SCRIPT)],
        "Downloading background clips",
    )

    print()
    print("Background download done.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print()
        print(f"Error: command failed with exit code {error.returncode}.")
        sys.exit(error.returncode)
    except Exception as error:
        print()
        print(f"Error: {error}")
        sys.exit(1)
