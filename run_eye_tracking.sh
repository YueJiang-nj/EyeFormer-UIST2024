#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --constraint=volta
#SBATCH --output=scanpath-prediction-population-rl.out

module load anaconda3
source activate eye_tracking

python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=6672 \
mse_tracking.py \
--output_dir output/scanpath_prediction_population_rl \
--checkpoint output/scanpath_prediction_population_pretrain/checkpoint_29.pth \

source deactivate
