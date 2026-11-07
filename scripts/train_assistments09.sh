#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
exec .venv/bin/python -u main.py --config configs/assistments09.yaml "$@"
