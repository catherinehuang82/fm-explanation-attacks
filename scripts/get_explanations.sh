#!/bin/bash
#SBATCH -J get_explanations # A single job name for the array
#SBATCH -n 1 # total number of processes
#SBATCH -N 1 # number of nodes
#SBATCH -p seas_compute,sapphire,shared # Partition
#SBATCH --mem 45000 # Memory request (45 GB)
#SBATCH -t 0-70:00 # (D-HH:MM)
#SBATCH -o /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/paper/attack_data/outputs/explanations/%j.out # Standard output
#SBATCH -e /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/paper/attack_data/errors/explanations/%j.err # Standard error
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=catherinehuang@college.harvard.edu

# just directly run the script!
conda run -n cnn2 python3 get_explanations.py --total_data_examples 20000 --explanation_type $1 --experiment_no $2 --epochs $3 --data $4 --model $5