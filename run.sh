#!/usr/bin/env bash
# Linux / macOS launcher: creates a venv, installs deps, trains if needed, starts the app.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN=${PYTHON_BIN:-python3}
[ -d venv ] || "$PYTHON_BIN" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
[ -f model.pkl ] && [ -f vectorizer.pkl ] || python train.py
echo "Open http://127.0.0.1:${PORT:-5000}"
python app.py
