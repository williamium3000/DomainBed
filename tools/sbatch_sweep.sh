#!/bin/bash -l
#SBATCH --job-name=sweep
#SBATCH --time=24:0:0
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

algorithm=ERM
dataset=PACS
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
       --hparams "$(<configs/${algorithm}/hparams.json)"\
       --output_dir "work_dirs/sweep/${algorithm}/${dataset}" 2>&1 | tee work_dirs/sweep/${algorithm}/${dataset}/$now.txt
