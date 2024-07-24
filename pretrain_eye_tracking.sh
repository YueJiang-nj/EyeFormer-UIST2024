#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --constraint=volta
#SBATCH --output=scanpath-prediction-population-pretrain.out

module load anaconda3
source activate eye_tracking

python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=7778 \
mse_tracking_pretrain.py \
--output_dir output/scanpath_prediction_population_pretrain \

source deactivate
