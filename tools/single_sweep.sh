dataset=$1
algorithms=$2
save_path=work_dirs/sweep/$dataset/$algorithms

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m domainbed.scripts.sweep delete_incomplete\
       --datasets ${dataset}\
       --algorithms $algorithms \
       --data_dir ./domainbed/data\
       --command_launcher multi_gpu\
       --single_test_envs\
       --steps 5001\
       --holdout_fraction 0.2\
       --n_hparams 5\
       --n_trials 3\
       --skip_confirmation\
       --hparams "$(<domainbed/configs/no_dropout.json)"\
       --output_dir $save_path
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m domainbed.scripts.sweep launch\
       --datasets ${dataset}\
       --algorithms $algorithms \
       --data_dir ./domainbed/data\
       --command_launcher multi_gpu\
       --single_test_envs\
       --steps 5001\
       --holdout_fraction 0.2\
       --n_hparams 5\
       --n_trials 3\
       --skip_confirmation\
       --hparams "$(<domainbed/configs/no_dropout.json)"\
       --output_dir $save_path