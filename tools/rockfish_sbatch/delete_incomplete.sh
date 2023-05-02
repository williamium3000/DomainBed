algorithm=$1
dataset=$2
output_dir=$3
python -m domainbed.scripts.sweep delete_incomplete\
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
       --output_dir $output_dir