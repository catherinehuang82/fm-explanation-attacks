#!/bin/bash
#SBATCH -J get_losses # A single job name for the array
#SBATCH -n 1 # total number of processes
#SBATCH -N 1 # number of nodes
#SBATCH -p seas_compute,sapphire,shared # Partition
#SBATCH --mem 45000 # Memory request
#SBATCH -t 0-40:00 # (D-HH:MM)
#SBATCH -o /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/paper/attack_data/outputs/losses/%j.out # Standard output
#SBATCH -e /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/paper/attack_data/errors/losses/%j.err # Standard error
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=catherinehuang@college.harvard.edu

# just directly run the script!
conda run -n cnn2 python3 get_losses.py --total_data_examples 20000 --experiment_no $1 --clipping_mode $2 --epsilon $3 --epochs $4 --data $5 --model $6