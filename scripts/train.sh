#!/bin/bash
#SBATCH -J train # A single job name for the array
#SBATCH --gres=gpu:1
#SBATCH -p seas_gpu # Partition
#SBATCH --mem 50000 # Memory request (30 GB)
#SBATCH -t 0-30:00 # (D-HH:MM)
#SBATCH -o /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/paper/attack_data/outputs/train/%j.out # Standard output
#SBATCH -e /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/paper/attack_data/errors/train/%j.err # Standard error
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=catherinehuang@college.harvard.edu

# just directly run the script!
conda run -n cnn2 python3 train.py --data CIFAR10 --total_data_examples 20000 --experiment_no $1 --clipping_mode $2 --epochs $3 --model $4