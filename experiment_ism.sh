#!/bin/bash
#SBATCH --job-name=ism_new_norm
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

cd "$SLURM_SUBMIT_DIR"

echo "Activating conda environment..."
source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda activate ism_env

echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

PYTHON=python
DATA="data/02_TUB_data.pth"

ROOT="runs/new_normalization"
COMMON="$ROOT/common"
OUT="$ROOT/baseline"

CROP_SIZE=512
NUM_ITER=1000
SAVE_EVERY=100
PRINT_EVERY=100

LR_X=1e-2
LR_K=1e-4

# Important: first run after changing Poisson sum -> mean
SMOOTH_K=0

PRETRAIN_ITER=50
PRETRAIN_LR=1e-4

SEED=0
Z_IDX=1
MID_NUM_ITER=15

mkdir -p "$COMMON"
mkdir -p "$OUT"


# ============================================================
# 1. Prepare common measurement / PSFs / MID
# ============================================================

if [[ ! -f "$COMMON/theory_mid.npz" ]]; then
    echo "Preparing theory/MID data..."

    "$PYTHON" compare_methods.py \
        --stage theory \
        --work_dir "$COMMON" \
        --path_data "$DATA" \
        --crop_size "$CROP_SIZE" \
        --z_in_idx "$Z_IDX" \
        --mid_num_iter "$MID_NUM_ITER"
else
    echo "Reusing $COMMON/theory_mid.npz"
fi

ln -sf "$(realpath "$COMMON/theory_mid.npz")" "$OUT/theory_mid.npz"


# ============================================================
# 2. Fixed theoretical-PSF DIP FIRST
# ============================================================

echo
echo "============================================================"
echo "FIXED THEORETICAL PSF DIP"
echo "============================================================"

"$PYTHON" compare_methods.py \
    --stage dip \
    --work_dir "$OUT" \
    --num_iter "$NUM_ITER" \
    --lr_x "$LR_X" \
    --lr_k "$LR_K" \
    --smooth_k "$SMOOTH_K" \
    --save_every "$SAVE_EVERY" \
    --print_every "$PRINT_EVERY" \
    --seed "$SEED" \
    --skip_blind \
    2>&1 | tee "$OUT/fixed.log"


# ============================================================
# 3. Blind DIP
# ============================================================

echo
echo "============================================================"
echo "BLIND DIP"
echo "lr_k=$LR_K | smooth_k=$SMOOTH_K"
echo "============================================================"

"$PYTHON" compare_methods.py \
    --stage dip \
    --work_dir "$OUT" \
    --num_iter "$NUM_ITER" \
    --lr_x "$LR_X" \
    --lr_k "$LR_K" \
    --smooth_k "$SMOOTH_K" \
    --pretraining \
    --pretraining_iter "$PRETRAIN_ITER" \
    --pretraining_lr "$PRETRAIN_LR" \
    --save_every "$SAVE_EVERY" \
    --print_every "$PRINT_EVERY" \
    --seed "$SEED" \
    --skip_fixed \
    2>&1 | tee "$OUT/blind.log"


# ============================================================
# 4. Blind-DIP evolution
# ============================================================

"$PYTHON" plot_dip_history.py \
    "$OUT/dip_blind.npz" \
    --out_dir "$OUT"


# ============================================================
# 5. Final comparison
# ============================================================

"$PYTHON" compare_methods.py \
    --stage figure \
    --work_dir "$OUT" \
    --psf_global_norm


echo
echo "============================================================"
echo "Finished."
echo "Results: $OUT"
echo "============================================================"