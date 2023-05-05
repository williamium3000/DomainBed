dataset=$1
algorithms=$2
save_path=work_dirs/single_run/vit-b16/$dataset/${algorithms}

mkdir -p $save_path

python3 -m domainbed.scripts.train \
       --data_dir=./domainbed/data \
       --algorithm $algorithms\
       --dataset $dataset \
       --test_envs 0 \
       --holdout_fraction 0.2\
       --steps 5001 \
       --hparams "$(<configs/clip_ft_vit-b16.json)" \
       --output_dir $save_path