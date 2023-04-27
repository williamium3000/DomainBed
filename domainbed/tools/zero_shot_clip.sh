dataset=$1
algorithms=CLIP
save_path=work_dirs/sweep/$dataset/$algorithms/RN50

python -m domainbed.scripts.sweep delete_incomplete\
       --datasets ${dataset}\
       --algorithms $algorithms \
       --data_dir ./domainbed/data\
       --command_launcher local\
       --single_test_envs\
       --steps 0\
       --holdout_fraction 0.2\
       --n_hparams 1\
       --n_trials 1\
       --skip_confirmation\
       --hparams "$(<domainbed/configs/clip_lp.json)"\
       --output_dir $save_path

python -m domainbed.scripts.sweep launch\
       --datasets ${dataset}\
       --algorithms $algorithms \
       --data_dir ./domainbed/data\
       --command_launcher local\
       --single_test_envs\
       --steps 0\
       --holdout_fraction 0.2\
       --n_hparams 1\
       --n_trials 1\
       --skip_confirmation\
       --hparams "$(<domainbed/configs/clip_lp.json)"\
       --output_dir $save_path