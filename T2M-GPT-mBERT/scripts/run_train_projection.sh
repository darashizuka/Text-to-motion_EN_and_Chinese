#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/scratch/xy68/NLP/T2M-GPT
VENV_DIR="$PROJECT_DIR/.venv-gpu"

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
import torch
print(torch.cuda.is_available())
assert torch.cuda.is_available()
PY
then
  echo "[run] CUDA is not available in current shell."
  exit 1
fi

python scripts/train_projection.py \
  --data-root dataset/HumanML3D \
  --split-file all.txt \
  --batch-size 128 \
  --epochs 20 \
  --lr 1e-4 \
  --out-dir output_projection \
  --exp-name HML3D_mBERT_proj
