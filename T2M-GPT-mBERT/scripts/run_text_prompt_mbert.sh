#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/scratch/xy68/NLP/T2M-GPT
VENV_DIR="$PROJECT_DIR/.venv-gpu"
HML3D_LAST2_PROJ_DIR="$PROJECT_DIR/output_projection/HML3D_mBERT_last2_proj"
HML3D_PROJ_DIR="$PROJECT_DIR/output_projection/HML3D_mBERT_proj"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_text_prompt_mbert.sh \"a person walks and waves\""
  echo "[run] checkpoint is auto-loaded from output_projection/HML3D_mBERT_last2_proj first"
  exit 1
fi

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

if ! python - <<'PY'
import importlib
importlib.import_module('transformers')
print('transformers OK')
PY
then
  echo "[run] transformers is missing. Install with: pip install transformers"
  exit 1
fi

if [[ -f "$HML3D_LAST2_PROJ_DIR/proj_best.pth" ]]; then
  PROJ_CKPT="$HML3D_LAST2_PROJ_DIR/proj_best.pth"
elif [[ -f "$HML3D_LAST2_PROJ_DIR/proj_last.pth" ]]; then
  PROJ_CKPT="$HML3D_LAST2_PROJ_DIR/proj_last.pth"
elif [[ -f "$HML3D_PROJ_DIR/proj_best.pth" ]]; then
  PROJ_CKPT="$HML3D_PROJ_DIR/proj_best.pth"
elif [[ -f "$HML3D_PROJ_DIR/proj_last.pth" ]]; then
  PROJ_CKPT="$HML3D_PROJ_DIR/proj_last.pth"
else
  echo "[run] HumanML3D checkpoint not found under:"
  echo "[run]   $HML3D_LAST2_PROJ_DIR"
  echo "[run]   $HML3D_PROJ_DIR"
  echo "[run] expected one of: proj_best.pth / proj_last.pth"
  exit 1
fi

echo "[run] using projection checkpoint: $PROJ_CKPT"

python scripts/text2motion_prompt_mbert.py --text "$*" --proj-ckpt "$PROJ_CKPT" --out-dir demo_outputs
