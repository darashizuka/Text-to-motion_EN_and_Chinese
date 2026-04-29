#!/bin/bash
# Usage: ./launch.sh train_mt5.py

SCRIPT_TO_RUN=$1
source config.sh

REMOTE_WORK_DIR="/storage/hpc/work/comp584/sd205"
REMOTE_SCRATCH_DIR="/scratch/sd205/comp584"
REMOTE_T2M_DIR="$REMOTE_SCRATCH_DIR/T2M-GPT"

echo "Uploading scripts to SCRATCH T2M-GPT..."
ssh $REMOTE "mkdir -p $REMOTE_T2M_DIR"
scp submit.slurm "$REMOTE:$REMOTE_WORK_DIR/submit.slurm"
scp mt5_encoder.py eval_trans_mt5.py "$SCRIPT_TO_RUN" "$REMOTE:$REMOTE_T2M_DIR/"

echo "Submitting job..."
ssh $REMOTE "cd ~ && sbatch --export=ALL,PYTHON_SCRIPT=$SCRIPT_TO_RUN $REMOTE_WORK_DIR/submit.slurm"
