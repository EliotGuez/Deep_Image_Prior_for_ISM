#!/bin/bash
#SBATCH --job-name=experiments_ism_02/04
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00


echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ism_env

echo "Conda environment activated: $CONDA_DEFAULT_ENV"

echo "Starting experiments..."

python DIP_ISM.py --phantom_types mixture  --save_path "test_ism/report_pret_psf/mixture_pret_psf0.5"  --pretraining --pret_psf 0.5
python DIP_ISM.py --phantom_types mixture  --save_path "test_ism/report_pret_psf/mixture_pret_psf1"  --pretraining --pret_psf 1.0
python DIP_ISM.py --phantom_types mixture  --save_path "test_ism/report_pret_psf/mixture_pret_psf1.5"  --pretraining --pret_psf 1.5
python DIP_ISM.py --phantom_types mixture  --save_path "test_ism/report_pret_psf/mixture_pret_psf2"  --pretraining --pret_psf 2.0
python DIP_ISM.py --phantom_types mixture  --save_path "test_ism/report_pret_psf/mixture_pret_psf3"  --pretraining --pret_psf 3.0
python DIP_ISM.py --phantom_types mixture  --save_path "test_ism/report_pret_psf/mixture_pret_psf4"  --pretraining --pret_psf 4.0
