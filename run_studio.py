#!/usr/bin/env python3
"""Launch Shakespeare Studio — FastAPI backend + Vite React frontend."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "backend"
FRONTEND_DIR = REPO_ROOT / "frontend"


def ensure_frontend_env() -> None:
    env_path = FRONTEND_DIR / ".env"
    example = FRONTEND_DIR / ".env.example"
    if not env_path.exists() and example.exists():
        shutil.copy(example, env_path)
        print(f"Created {env_path.relative_to(REPO_ROOT)} from .env.example")


def ensure_frontend_deps(skip_install: bool) -> None:
    if skip_install or (FRONTEND_DIR / "node_modules").exists():
        return
    print("Installing frontend dependencies (npm install)…")
    subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)


def wait_for_processes(processes: list[subprocess.Popen]) -> None:
    try:
        while True:
            for proc in processes:
                code = proc.poll()
                if code is not None:
                    names = ["backend", "frontend"]
                    idx = processes.index(proc)
                    raise SystemExit(
                        f"{names[idx]} exited unexpectedly with code {code}."
                    )
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down Shakespeare Studio…")


def terminate_processes(processes: list[subprocess.Popen]) -> None:
    for proc in processes:
        if proc.poll() is not None:
            continue
        proc.terminate()
    deadline = time.time() + 5
    for proc in processes:
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if proc.poll() is None:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Shakespeare Studio backend and frontend together.",
    )
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=5173)
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip automatic npm install when node_modules is missing",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable uvicorn auto-reload",
    )
    args = parser.parse_args()

    if not BACKEND_DIR.is_dir():
        sys.exit(f"Missing backend directory: {BACKEND_DIR}")
    if not FRONTEND_DIR.is_dir():
        sys.exit(f"Missing frontend directory: {FRONTEND_DIR}")
    if shutil.which("npm") is None:
        sys.exit("npm not found. Install Node.js to run the React frontend.")

    try:
        import uvicorn  # noqa: F401
    except ImportError:
        sys.exit(
            "uvicorn not installed. Run: pip install -r backend/requirements.txt"
        )

    ensure_frontend_env()
    ensure_frontend_deps(args.no_install)

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.backend_port),
    ]
    if not args.no_reload:
        backend_cmd.append("--reload")

    frontend_env = os.environ.copy()
    frontend_env.setdefault(
        "VITE_API_BASE_URL",
        f"http://127.0.0.1:{args.backend_port}",
    )

    frontend_cmd = [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.frontend_port),
    ]

    print("Starting Shakespeare Studio")
    print(f"  Backend:  http://127.0.0.1:{args.backend_port}")
    print(f"  Frontend: http://127.0.0.1:{args.frontend_port}")
    print("  Press Ctrl+C to stop both servers.\n")

    backend = subprocess.Popen(
        backend_cmd,
        cwd=BACKEND_DIR,
        env=os.environ.copy(),
    )
    frontend = subprocess.Popen(
        frontend_cmd,
        cwd=FRONTEND_DIR,
        env=frontend_env,
    )

    processes = [backend, frontend]

    def handle_signal(signum, _frame):
        terminate_processes(processes)
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        wait_for_processes(processes)
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    main()
