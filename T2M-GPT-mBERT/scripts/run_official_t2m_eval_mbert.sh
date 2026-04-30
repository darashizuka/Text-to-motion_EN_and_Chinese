#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/scratch/xy68/NLP/T2M-GPT
VENV_DIR="$PROJECT_DIR/.venv-gpu"
RUN_TAG=${1:-OFFICIAL_GPT_MBERT_$(date +%Y%m%d_%H%M%S)}
# Optional explicit proj checkpoint path.
MBERT_PROJ_CKPT=${2:-}

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

for rel in \
  glove \
  checkpoints/t2m/Comp_v6_KLD005/opt.txt \
  checkpoints/t2m/text_mot_match/model/finest.tar \
  pretrained/VQVAE/net_last.pth \
  pretrained/VQTransformer_corruption05/net_best_fid.pth
 do
  if [[ ! -e "$PROJECT_DIR/$rel" ]]; then
    echo "[run] missing required path: $PROJECT_DIR/$rel"
    exit 1
  fi
 done

PROJ_ARGS=()
if [[ -n "$MBERT_PROJ_CKPT" ]]; then
  if [[ ! -f "$MBERT_PROJ_CKPT" ]]; then
    echo "[run] mbert projection checkpoint not found: $MBERT_PROJ_CKPT"
    exit 1
  fi
  PROJ_ARGS=(--mbert-proj-ckpt "$MBERT_PROJ_CKPT")
fi

mkdir -p output_eval

echo "[run] HumanML3D mBERT official-like evaluation"
echo "[run] exp_name: $RUN_TAG"
echo "[run] repeat_time: 1"
if [[ -n "$MBERT_PROJ_CKPT" ]]; then
  echo "[run] using explicit mBERT proj ckpt: $MBERT_PROJ_CKPT"
else
  echo "[run] mBERT proj ckpt: auto-resolve from output_projection/*"
fi

python GPT_eval_multi_mbert.py \
  --dataname t2m \
  --embed-dim-gpt 1024 \
  --num-layers 9 \
  --n-head-gpt 16 \
  --block-size 51 \
  --ff-rate 4 \
  --resume-pth pretrained/VQVAE/net_last.pth \
  --resume-trans pretrained/VQTransformer_corruption05/net_best_fid.pth \
  --out-dir output_eval \
  --exp-name "$RUN_TAG" \
  --vq-name VQVAE \
  --device cuda \
  --repeat-time 1 \
  "${PROJ_ARGS[@]}"

echo "[run] done"
echo "[run] output dir: output_eval/$RUN_TAG"
