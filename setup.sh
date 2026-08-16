#!/usr/bin/env bash
# Create the inference environment. Requires uv (https://docs.astral.sh/uv/).
set -eu
cd "$(dirname "$0")"
uv venv --python 3.11 .venv
VIRTUAL_ENV="$PWD/.venv" uv pip install -r requirements.txt
echo "done — run inference with: .venv/bin/python restore.py in.wav out.wav"
