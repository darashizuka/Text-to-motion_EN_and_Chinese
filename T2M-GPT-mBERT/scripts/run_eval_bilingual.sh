#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/scratch/xy68/NLP/T2M-GPT
VENV_DIR="$PROJECT_DIR/.venv-gpu"
HML3D_ROOT="$PROJECT_DIR/dataset/HumanML3D"
SPLIT_FILE="test.txt"
SAMPLE_CSV="$PROJECT_DIR/dataset/HumanML3D/test_bilingual_eval_100.csv"
RUN_TAG=${1:-HML3D_TEST100_EN_ZH_$(date +%Y%m%d_%H%M%S)}
OUT_DIR="$PROJECT_DIR/output_eval_bilingual/$RUN_TAG"

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

if [[ ! -d "$HML3D_ROOT" ]]; then
  echo "[run] missing HumanML3D root: $HML3D_ROOT"
  exit 1
fi

if [[ ! -f "$HML3D_ROOT/$SPLIT_FILE" ]]; then
  echo "[run] missing split file: $HML3D_ROOT/$SPLIT_FILE"
  exit 1
fi

source "$VENV_DIR/bin/activate"

if ! python - <<'PY'
import importlib

missing = []
for name in ("transformers", "sentencepiece"):
  try:
    importlib.import_module(name)
  except Exception:
    missing.append(name)

if missing:
  raise SystemExit("MISSING:" + ",".join(missing))
print("deps_ok")
PY
then
  echo "[run] missing dependency for EN->ZH translation."
  echo "[run] please install in venv: pip install transformers sentencepiece"
  exit 1
fi

if ! python - <<'PY'
import torch
print(torch.cuda.is_available())
assert torch.cuda.is_available()
PY
then
  echo "[run] CUDA is not available in current shell."
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "[run] out_dir: $OUT_DIR"
echo "[run] HumanML3D root: $HML3D_ROOT"
echo "[run] split file: $SPLIT_FILE"
echo "[run] fixed sample csv: $SAMPLE_CSV"
echo "[run] prompts: 100 (EN sampled from test, ZH auto-translated)"
echo "[run] methods: mbert + clip (standard baseline: clip)"

python scripts/eval_bilingual.py \
  --hml3d-root "$HML3D_ROOT" \
  --split-file "$SPLIT_FILE" \
  --sample-csv "$SAMPLE_CSV" \
  --max-pairs 100 \
  --shuffle-samples \
  --out-dir "$OUT_DIR" \
  --device cuda \
  --methods mbert clip \
  --comparison-standard clip \
  --save-motion-npy

echo "[run] done"
echo "[run] summary csv: $OUT_DIR/evaluation_summary.csv"
echo "[run] model comparison csv: $OUT_DIR/model_comparison.csv"
echo "[run] gt reference pair csv: $OUT_DIR/gt_reference_pair_metrics.csv"
echo "[run] pair metrics csv: $OUT_DIR/pair_metrics.csv"
echo "[run] prompt outputs csv: $OUT_DIR/prompt_outputs.csv"
