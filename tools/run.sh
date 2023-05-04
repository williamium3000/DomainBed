python3 -m domainbed.scripts.train \
       --data_dir=./domainbed/data \
       --algorithm LanguageDrivenDGV2\
       --dataset PACSWithDomain \
       --steps 5001 \
       --hparams "$(<configs/clip_ft.json)"