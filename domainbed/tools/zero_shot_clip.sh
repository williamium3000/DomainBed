dataset=$1
algorithms=CLIP
save_path=work_dirs/sweep/$dataset/$algorithms/ViT_B-32

python -m domainbed.scripts.sweep delete_incomplete\
       --datasets ${dataset}\
       --algorithms $algorithms \
       --data_dir ./domainbed/data\
       --command_launcher local\
       --single_test_envs\
       --steps 1\
       --holdout_fraction 0.2\
       --n_hparams 1\
       --n_trials 1\
       --skip_confirmation\
       --hparams "$(<domainbed/configs/clip_zeroshot.json)"\
       --output_dir $save_path

python -m domainbed.scripts.sweep launch\
       --datasets ${dataset}\
       --algorithms $algorithms \
       --data_dir ./domainbed/data\
       --command_launcher local\
       --single_test_envs\
       --steps 1\
       --holdout_fraction 0.2\
       --n_hparams 1\
       --n_trials 1\
       --skip_confirmation\
       --hparams "$(<domainbed/configs/clip_zeroshot.json)"\
       --output_dir $save_path