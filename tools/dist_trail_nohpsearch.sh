dataset=$1
algorithms=$2
save_path=work_dirs/sweep/vit-b16/${dataset}_extra/${algorithms}

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
       --hparams "$(<configs/clipood_vit-b16_with_extra.json)"\
       --output_dir $save_path
CUDA_VISIBLE_DEVICES=3,4,5 python -m domainbed.scripts.sweep launch\
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
       --hparams "$(<configs/clipood_vit-b16_with_extra.json)"\
       --output_dir $save_path