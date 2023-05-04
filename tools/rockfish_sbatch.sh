algorithm=ERM_CLIP_Logits_EMA
dataset=PACS

save_path=work_dirs/sweep/$dataset/$algorithms

bash tools/rockfish_sbatch/delete_incomplete.sh $algorithm $dataset $save_path
sbatch tools/rockfish_sbatch/sbatch_sweep.sh $algorithm $dataset $save_path