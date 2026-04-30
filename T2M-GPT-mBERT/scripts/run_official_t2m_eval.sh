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

HAS_T2M=0
HAS_KIT=0
if [[ -d dataset/HumanML3D/new_joint_vecs && -d dataset/HumanML3D/texts ]]; then
  HAS_T2M=1
fi
if [[ -d dataset/KIT-ML/new_joint_vecs && -d dataset/KIT-ML/texts ]]; then
  HAS_KIT=1
fi

echo "[run] host: $(hostname)"
echo "[run] cuda: $(python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
)"

if [[ "$HAS_T2M" -eq 1 ]]; then
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

  mkdir -p output_eval
  echo "[run] HumanML3D detected, running full official GPT evaluation"
  python GPT_eval_multi.py \
    --dataname t2m \
    --embed-dim-gpt 1024 \
    --num-layers 9 \
    --n-head-gpt 16 \
    --block-size 51 \
    --ff-rate 4 \
    --resume-pth pretrained/VQVAE/net_last.pth \
    --resume-trans pretrained/VQTransformer_corruption05/net_best_fid.pth \
    --out-dir output_eval \
    --exp-name OFFICIAL_GPT \
    --vq-name VQVAE
elif [[ "$HAS_KIT" -eq 1 ]]; then
  for rel in \
    dataset/KIT-ML/new_joint_vecs \
    dataset/KIT-ML/texts \
    checkpoints/kit/Comp_v6_KLD005/opt.txt
  do
    if [[ ! -e "$PROJECT_DIR/$rel" ]]; then
      echo "[run] missing required path: $PROJECT_DIR/$rel"
      exit 1
    fi
  done

  echo "[run] KIT-ML detected, running KIT GPU smoke/eval check"
  echo "[run] note: official pretrained in pretrained/ are t2m-only, not KIT-specific"
  python - <<'PY'
import torch
from options import option_vq
from dataset import dataset_VQ
import models.vqvae as vqvae

assert torch.cuda.is_available(), "CUDA is not available"

args = option_vq.get_args_parser()
args.dataname = "kit"
args.quantizer = "orig"

loader = dataset_VQ.DATALoader(
    "kit",
    batch_size=4,
    window_size=args.window_size,
    unit_length=2**args.down_t,
)

x = next(iter(loader)).cuda().float()

net = vqvae.HumanVQVAE(
    args,
    args.nb_code,
    args.code_dim,
    args.output_emb_width,
    args.down_t,
    args.stride_t,
    args.width,
    args.depth,
    args.dilation_growth_rate,
    args.vq_act,
    args.vq_norm,
).cuda().eval()

with torch.no_grad():
    y, loss, ppl = net(x)

print("KIT_SMOKE_OK")
print("batch_shape:", tuple(x.shape))
print("output_shape:", tuple(y.shape))
print("commit_loss:", float(loss))
print("perplexity:", float(ppl))
PY
else
  echo "[run] neither HumanML3D nor KIT-ML dataset found"
  echo "[run] expected one of: dataset/HumanML3D or dataset/KIT-ML"
  exit 1
fi
