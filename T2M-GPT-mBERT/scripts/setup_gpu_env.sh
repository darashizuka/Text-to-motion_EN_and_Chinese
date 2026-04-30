#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/scratch/xy68/NLP/T2M-GPT}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv-gpu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "[setup] reusing virtual environment at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install --upgrade scipy scikit-learn tensorboard tqdm pyyaml matplotlib pandas imageio moviepy gdown ftfy regex einops ipdb smplx transformers sentencepiece

if command -v git >/dev/null 2>&1; then
  python -m pip install --upgrade git+https://github.com/openai/CLIP.git
else
  echo "[setup] git not found, installing CLIP from GitHub source zip"
  python -m pip install --upgrade "clip @ https://github.com/openai/CLIP/archive/refs/heads/main.zip"
fi

# Download helper assets only when missing.
if [[ ! -d glove ]]; then
  bash dataset/prepare/download_glove.sh
fi

if [[ ! -d checkpoints/kit || ! -d checkpoints/t2m ]]; then
  bash dataset/prepare/download_extractor.sh
fi

if [[ ! -d pretrained/VQVAE || ! -d pretrained/VQTransformer_corruption05 ]]; then
  bash dataset/prepare/download_model.sh
fi

# KIT-ML is publicly downloadable from HumanML3D repo Google Drive folder.
if [[ ! -d dataset/KIT-ML/new_joint_vecs || ! -d dataset/KIT-ML/texts ]]; then
  mkdir -p dataset
  gdown --folder --remaining-ok "https://drive.google.com/drive/folders/1D3bf2G2o4Hv-Ale26YW18r1Wrh7oIAwK?usp=sharing" -O dataset/

  mkdir -p .tools/7zip
  cd .tools/7zip
  if [[ ! -x ./7zz ]]; then
    wget -q https://www.7-zip.org/a/7z2501-linux-x64.tar.xz -O 7z.tar.xz
    tar -xf 7z.tar.xz
    chmod +x 7zz
  fi
  cd "$PROJECT_DIR/dataset/KIT-ML"
  ../../.tools/7zip/7zz x -y new_joint_vecs.rar
  ../../.tools/7zip/7zz x -y texts.rar
  cd "$PROJECT_DIR"
fi

touch "$VENV_DIR/.t2mgpt_ready"

python - <<'PY'
import torch
import clip
print("[setup] torch:", torch.__version__)
print("[setup] cuda_available_now:", torch.cuda.is_available())
print("[setup] clip_ok:", hasattr(clip, "tokenize"))
PY

echo "[setup] done"
