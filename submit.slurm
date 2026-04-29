#!/bin/bash
#SBATCH --job-name=comp584_t2m
#SBATCH --partition=scavenge
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=00:59:00
#SBATCH --output=/scratch/sd205/comp584/logs/%x_%j.out
#SBATCH --error=/scratch/sd205/comp584/logs/%x_%j.err

echo "Job started on $(hostname) at $(date)"
nvidia-smi

# === Paths ===
WORK_DIR="/storage/hpc/work/comp584/sd205"
SCRATCH_DIR="/scratch/sd205/comp584"

# === Copy from WORK to SCRATCH (only needed on first run or if code changed) ===
# This runs on the COMPUTE node which can see /scratch but not /storage/hpc/work
# So we do the copy from the login node BEFORE sbatch, via launch.sh
# Here we just verify the files exist on scratch

echo "Checking scratch directory..."
if [ ! -d "$SCRATCH_DIR/T2M-GPT" ]; then
    echo "ERROR: T2M-GPT not found on scratch. Run setup_scratch.sh first."
    exit 1
fi

# === Setup ===
export HF_HOME="$SCRATCH_DIR/hf_cache"
mkdir -p "$HF_HOME"

# Load Python (no module purge - keep defaults)
module load Python/3.11.5 2>/dev/null || module load python 2>/dev/null || true

# Activate venv
source "$SCRATCH_DIR/t2m-env/bin/activate"

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# === Run ===
cd "$SCRATCH_DIR/T2M-GPT"
srun python "$PYTHON_SCRIPT" \
    --dataname t2m \
    --resume-pth ./pretrained/VQVAE/net_best_fid.pth \
    --exp-name mt5_t2m \
    --resume-trans ./output_GPT_Final/mt5_t2m/net_best_fid.pth \
    --down-t 2 \
    --depth 3 \
    --dilation-growth-rate 3 \
    --vq-act relu \
    --block-size 51 \
    --total-iter 15000 \
    --eval-iter 10000 \
    --print-iter 500

echo "Job finished at $(date)"
