import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

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
        print(f"Virtuelle Umgebung gefunden: {VENV_DIR}", flush=True)
        return python_path

    print(f"Erstelle virtuelle Umgebung: {VENV_DIR}", flush=True)
    venv.create(VENV_DIR, with_pip=True)

    if not python_path.exists():
        raise RuntimeError(f"Python in der virtuellen Umgebung nicht gefunden: {python_path}")

    return python_path


def install_requirements(python_path: Path) -> None:
    if not REQUIREMENTS_FILE.exists():
        print("Keine requirements.txt gefunden, ueberspringe Installation.", flush=True)
        return

    run(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS_FILE),
        ],
        "Installiere requirements",
    )


def run_pipeline(python_path: Path) -> None:
    for script in PIPELINE_SCRIPTS:
        if not script.exists():
            raise FileNotFoundError(f"Skript nicht gefunden: {script}")

        run([str(python_path), str(script)], f"Starte {script.relative_to(PROJECT_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erstellt .venv, installiert requirements und startet die Audio-Pipeline."
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Nur .venv erstellen und requirements installieren.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Requirements nicht installieren.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.setup_only and sys.version_info >= (3, 11):
        raise RuntimeError(
            "Die volle OpenWakeWord-Trainingspipeline ist auf Python 3.10 ausgelegt. "
            "Bitte auf der Trainingsmaschine mit Python 3.10 starten, z.B. `python3.10 main.py`."
        )

    python_path = ensure_venv()

    if not args.skip_install:
        install_requirements(python_path)

    if args.setup_only:
        print(flush=True)
        print("Setup fertig.", flush=True)
        return

    run_pipeline(python_path)

    print(flush=True)
    print("Pipeline fertig.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(flush=True)
        print(f"Fehler: Befehl fehlgeschlagen mit Code {error.returncode}.", flush=True)
        sys.exit(error.returncode)
    except Exception as error:
        print(flush=True)
        print(f"Fehler: {error}", flush=True)
        sys.exit(1)
