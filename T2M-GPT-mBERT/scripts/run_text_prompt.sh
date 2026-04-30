#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/scratch/xy68/NLP/T2M-GPT
VENV_DIR="$PROJECT_DIR/.venv-gpu"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_text_prompt.sh \"a person walks and waves\""
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

python scripts/text2motion_prompt.py --text "$*" --out-dir demo_outputs
