dataset=$1
algorithms=$2
save_path=work_dirs/sweep/vit-b16/$dataset/${algorithms}

python -m domainbed.scripts.sweep delete_incomplete\
       --datasets ${dataset}\
       --algorithms $algorithms \
       --data_dir ./domainbed/data\
       --command_launcher multi_gpu\
       --single_test_envs\
       --steps 5001\
       --holdout_fraction 0.2\
       --n_hparams 1\
       --n_trials 3\
       --skip_confirmation\
       --hparams "$(<configs/clipood_vit-b16.json)"\
       --output_dir $save_path
srun --partition=gpuA40x4 \
    --gres=gpu:2 \
    --ntasks-per-node=1 \
    --ntasks=1 \
    --job-name=sweep \
    --mem-per-cpu=16GB --cpus-per-task=3 \
    --time 02-00:00:00 \
    -A bbrt-delta-gpu   \
    --kill-on-bad-exit=1 \
    python -m domainbed.scripts.sweep launch\
       --datasets ${dataset}\
       --algorithms $algorithms \
       --data_dir ./domainbed/data\
       --command_launcher multi_gpu\
       --single_test_envs\
       --steps 5001\
       --holdout_fraction 0.2\
       --n_hparams 1\
       --n_trials 3\
       --skip_confirmation\
       --hparams "$(<configs/clipood_vit-b16.json)"\
       --output_dir $save_path