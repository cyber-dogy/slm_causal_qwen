#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Checking release tree: ${ROOT_DIR}"
echo

echo "[1/3] Local-only files"
find "${ROOT_DIR}" -maxdepth 3 \
  \( -path "${ROOT_DIR}/configs/*.local.yaml" -o -path "${ROOT_DIR}/configs/*.local.yml" -o -path "${ROOT_DIR}/configs/*.local.env" \) \
  -print || true
echo

echo "[2/3] Local traces"
find "${ROOT_DIR}" -maxdepth 4 \
  \( -name ".git" -o -name ".vscode" -o -name ".idea" -o -name "__pycache__" -o -name "*.pyc" -o -name "*.pyo" -o -name ".DS_Store" \) \
  -print || true
echo

echo "[3/3] Runtime/data/weight directories"
find "${ROOT_DIR}" -maxdepth 3 \
  \( -path "${ROOT_DIR}/outputs" -o -path "${ROOT_DIR}/wandb" -o -path "${ROOT_DIR}/paper_visualization/cache" -o -path "${ROOT_DIR}/paper_visualization/exports" -o -path "${ROOT_DIR}/data" -o -path "${ROOT_DIR}/datasets" -o -path "${ROOT_DIR}/model_cache" \) \
  -print || true
echo

echo "Done. Review the listed paths before publishing."
