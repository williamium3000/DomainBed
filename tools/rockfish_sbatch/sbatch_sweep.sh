#!/bin/bash -l
#SBATCH --job-name=sweep
#SBATCH --time=48:0:0
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH -A danielk_gpu
#SBATCH --output "slurm_logs/slurm-%j.out"


module load anaconda
module load imagemagick/7.1.0
conda activate pytorch2.0

algorithm=$1
dataset=$2
output_dir=$3
now=$(date +"%Y%m%d_%H%M%S")


srun --kill-on-bad-exit=1 python -m domainbed.scripts.sweep launch\
       --datasets ${dataset}\
       --algorithms ${algorithm} \
       --data_dir ./domainbed/data\
       --command_launcher local\
       --single_test_envs\
       --steps 5001\
       --holdout_fraction 0.2\
       --n_hparams 5\
       --n_trials 3\
       --skip_confirmation\
       --hparams "$(<configs/clip.json)"\
       --output_dir $output_dir 2>&1 | tee $output_dir/$now.txt
