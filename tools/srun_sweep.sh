algorithm=ERM
dataset=PACS
now=$(date +"%Y%m%d_%H%M%S")

srun --partition a100 \
    --gres=gpu:1 \
    --ntasks-per-node=1 \
    --ntasks=1 \
    --job-name=sweep \
    --mem=60G \
    --time 02-00:00:00 \
    -A danielk_gpu   \
    --kill-on-bad-exit=1 \
    python -m domainbed.scripts.sweep launch\
       --datasets ${dataset}\
       --algorithms ${algorithm} \
       --data_dir ./domainbed/data\
       --command_launcher local\
       --single_test_envs\
       --steps 5001\
       --holdout_fraction 0.2\
       --n_hparams 10\
       --n_trials 2\
       --skip_confirmation\
       --hparams "$(<configs/${algorithm}/hparams.json)"\
       --output_dir "work_dirs/sweep/${algorithm}/${dataset}" 2>&1 | tee work_dirs/sweep/${algorithm}/${dataset}/$now.txt
