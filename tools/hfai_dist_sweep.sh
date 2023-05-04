dataset=PACS
algorithms=ERM
save_path=work_dirs/sweep/vit-b16/$dataset/$algorithms

python -m domainbed.scripts.sweep delete_incomplete \
       --datasets ${dataset} \
       --algorithms ${algorithms} \
       --data_dir domainbed/data \
       --command_launcher multi_gpu \
       --single_test_envs \
       --steps 5001 \
       --holdout_fraction 0.2 \
       --n_hparams 5 \
       --n_trials 3 \
       --skip_confirmation \
       --hparams "$(<configs/hfai_vit-b16_no_dropout.json)" \
       --output_dir ${save_path}
python -m domainbed.scripts.sweep launch \
       --datasets ${dataset} \
       --algorithms ${algorithms} \
       --data_dir domainbed/data \
       --command_launcher multi_gpu \
       --single_test_envs \
       --steps 5001\
       --holdout_fraction 0.2 \
       --n_hparams 5 \
       --n_trials 3 \
       --skip_confirmation \
       --hparams "$(<configs/hfai_vit-b16_no_dropout.json)" \
       --output_dir ${save_path}