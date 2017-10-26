#!/bin/bash
#
#SBATCH --job-name=super
#SBATCH --partition=p1080_4
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
#SBATCH --mem=20GB
#SBATCH --output=outputs/%A.out
#SBATCH --error=outputs/%A.err

module purge
module load python3/intel/3.5.3
module load pillow/python3.5/intel/4.2.1
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9

cd /scratch/jmw784/superresolution/superresolution-gan/

python -u train.py $1 > logs/$2.log
