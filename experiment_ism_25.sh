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



# python train_zernike_net.py  --mode train --save_path zk_net/center --batch_size 64 --num_steps 30000 --log_every 1000

python train_zernike_net.py --mode eval --checkpoint zk_net/center/last.pth  --batch_size 64 --eval_steps 25 --seed 10


# python DIP_ISM_2psf.py --calibration --save_path report_zk/testcenter/ --num_iter 400 --poisson_gain 0.4 --beads_sigma 1. --do_zernike --coeff_scale 0.4 --seed 10 


# python DIP_ISM_2psf.py --calibration --save_path report_zk/testNN2/iit05 --num_iter 400 --poisson_gain 0.4 --beads_sigma 1. --do_zernike --zernike_method iit --coeff_scale 0.5