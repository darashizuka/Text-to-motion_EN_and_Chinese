#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/scratch/xy68/NLP/T2M-GPT
VENV_DIR="$PROJECT_DIR/.venv-gpu"
BASE_PROJ_DIR="$PROJECT_DIR/output_projection/HML3D_mBERT_proj"

cd "$PROJECT_DIR"

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  set +u
  source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
  set -u
fi

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load CUDA/12.8.0 || true
  module load cuDNN/9.10.1.4-CUDA-12.8.0 || true
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[run] missing virtualenv: $VENV_DIR"
  echo "[run] please run: bash scripts/setup_gpu_env.sh"
  exit 1
fi

source "$VENV_DIR/bin/activate"

if ! python - <<'PY'
import importlib
import torch
for name in ("transformers", "clip"):
  importlib.import_module(name)
print(torch.cuda.is_available())
assert torch.cuda.is_available()
PY
then
  echo "[run] missing dependencies or CUDA is unavailable."
  exit 1
fi

INIT_CKPT=""
if [[ -f "$BASE_PROJ_DIR/proj_best.pth" ]]; then
  INIT_CKPT="$BASE_PROJ_DIR/proj_best.pth"
elif [[ -f "$BASE_PROJ_DIR/proj_last.pth" ]]; then
  INIT_CKPT="$BASE_PROJ_DIR/proj_last.pth"
fi

echo "[run] init checkpoint: ${INIT_CKPT:-none}"

CMD=(
  python scripts/train_mbert_last2_projection.py
  --data-root dataset/HumanML3D
  --split-file all.txt
  --batch-size 128
  --epochs 20
  --lr-proj 1e-4
  --lr-mbert 2e-5
  --train-last-n-layers 2
  --out-dir output_projection
  --exp-name HML3D_mBERT_last2_proj
)

if [[ -n "$INIT_CKPT" ]]; then
  CMD+=(--init-ckpt "$INIT_CKPT")
fi

"${CMD[@]}"
