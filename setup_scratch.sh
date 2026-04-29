#!/bin/bash
# Run this ON THE LOGIN NODE to copy everything from WORK to SCRATCH
# Usage: ssh sd205@nots.rice.edu 'bash /storage/hpc/work/comp584/sd205/setup_scratch.sh'

WORK_DIR="/storage/hpc/work/comp584/sd205"
SCRATCH_DIR="/scratch/sd205/comp584"

echo "Setting up scratch directory at $SCRATCH_DIR ..."
mkdir -p "$SCRATCH_DIR"

# Copy T2M-GPT codebase (including dataset symlink contents)
echo "Copying T2M-GPT..."
rsync -av --progress "$WORK_DIR/T2M-GPT/" "$SCRATCH_DIR/T2M-GPT/"

# Copy the virtual environment
echo "Copying venv..."
rsync -av "$WORK_DIR/t2m-env/" "$SCRATCH_DIR/t2m-env/"

# Fix venv paths (they contain hardcoded paths to the original location)
echo "Fixing venv paths..."
sed -i "s|$WORK_DIR/t2m-env|$SCRATCH_DIR/t2m-env|g" "$SCRATCH_DIR/t2m-env/bin/activate"
sed -i "s|$WORK_DIR/t2m-env|$SCRATCH_DIR/t2m-env|g" "$SCRATCH_DIR/t2m-env/bin/pip"
sed -i "s|$WORK_DIR/t2m-env|$SCRATCH_DIR/t2m-env|g" "$SCRATCH_DIR/t2m-env/pyvenv.cfg" 2>/dev/null

echo "Done! Scratch is ready at $SCRATCH_DIR"
echo "Contents:"
ls "$SCRATCH_DIR/"
ls "$SCRATCH_DIR/T2M-GPT/"
