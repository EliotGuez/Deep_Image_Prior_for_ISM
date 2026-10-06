#!/bin/bash
#SBATCH --job-name=ism_smooth_test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda activate ism_env

echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

PYTHON=python
DATA="data/02_TUB_data.pth"

ROOT="runs062"
COMMON="$ROOT/common"

CROP_SIZE=512
NUM_ITER=1000
MID_NUM_ITER=15
SAVE_EVERY=100
PRINT_EVERY=100

LR_X=1e-2
LR_K=1e-4

PRETRAIN_ITER=50
PRETRAIN_LR=1e-4
PRET_PSF=4.0

SEED=0
Z_IDX=1

# Fixed detector geometry
M=450
ROTATION_DEG=-76.30
MIRRORING=-1

mkdir -p "$COMMON"


# ============================================================
# 1. Measurement + MID
# ============================================================

"$PYTHON" compare_methods.py \
    --stage theory \
    --work_dir "$COMMON" \
    --path_data "$DATA" \
    --crop_size "$CROP_SIZE" \
    --z_in_idx "$Z_IDX" \
    --mid_planes all \
    --mid_num_iter "$MID_NUM_ITER"


# ============================================================
# 2. Fixed theoretical-PSF DIP
# Run only once because smooth_k does not affect this method
# ============================================================

"$PYTHON" compare_methods.py \
    --stage dip \
    --work_dir "$COMMON" \
    --num_iter "$NUM_ITER" \
    --lr_x "$LR_X" \
    --save_every "$SAVE_EVERY" \
    --print_every "$PRINT_EVERY" \
    --seed "$SEED" \
    --skip_blind \
    2>&1 | tee "$COMMON/fixed.log"


# ============================================================
# 3. Test different PSF regularization values
# ============================================================

for SMOOTH_K in 0 100 1000 10000 50000
do

    OUT="$ROOT/smooth_${SMOOTH_K}"
    mkdir -p "$OUT"

    echo
    echo "============================================================"
    echo "BLIND DIP"
    echo "smooth_k = $SMOOTH_K"
    echo "============================================================"

    # Reuse the same MID and fixed-DIP results
    ln -sf "$(realpath "$COMMON/theory_mid.npz")" \
        "$OUT/theory_mid.npz"

    ln -sf "$(realpath "$COMMON/dip_fixed.npz")" \
        "$OUT/dip_fixed.npz"


    # --------------------------------------------------------
    # Blind DIP
    # --------------------------------------------------------

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
        --pret_psf "$PRET_PSF" \
        --save_every "$SAVE_EVERY" \
        --print_every "$PRINT_EVERY" \
        --seed "$SEED" \
        --magnification "$M" \
        --rotation_deg "$ROTATION_DEG" \
        --mirroring "$MIRRORING" \
        --skip_fixed \
        2>&1 | tee "$OUT/blind.log"


    # --------------------------------------------------------
    # History of reconstruction + latent PSFs
    # --------------------------------------------------------

    "$PYTHON" plot_dip_history.py \
        "$OUT/dip_blind.npz" \
        --out_dir "$OUT"


    # --------------------------------------------------------
    # Final six-panel figure
    # --------------------------------------------------------

    "$PYTHON" compare_methods.py \
        --stage figure \
        --work_dir "$OUT"

done


echo
echo "============================================================"
echo "ALL REGULARIZATION TESTS FINISHED"
echo "Results in: $ROOT"
echo "============================================================"