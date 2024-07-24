# EyeFormer: Predicting Scanpaths in Free-Viewing Tasks with Transformer-Guided Reinforcement Learning

The implementation of the model in the submission **EyeFormer: Predicting Scanpaths in Free-Viewing Tasks with Transformer-Guided Reinforcement Learning**. 

## Requirement
```sh
conda create -n your_env python=3.6.13
conda activate your_env
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install transformers==4.8.1
pip install timm==0.4.9
conda install ruamel_yaml
pip install opencv-python
pip install --upgrade Pillow
pip install einops
pip install multimatch-gaze
```

The requirement is also uploaded as `requirements.txt`. Please use this file to make sure that some of the packages you have installed are the same. Please **don't** configure your environment with this file.

## Data Preparing
The dataset has the following file layout:
```
your_data_path/
|–– full/
|   |–– dataset/
|   |   |   |–– train/
|   |   |   |   |–– block 0/
|   |   |   |   |–– block 1/
|   |   |   |   |–– block 2/
|   |   |   |–– test/
|   |   |   |   |–– block 53/
|   |   |   |   |–– block 54/
|   |   |   |   |–– block 55/
|   |–– new_logs/
|   |   |–– kh000/
|   |   |–– kh001/
|   |   |–– kh002/
|   |–– saliency_maps
|   |   |–– block_0
|   |   |–– block_1
|   |   |–– block_2
```

You can download them [here](https://drive.google.com/drive/folders/1Qs5YtCegqz6sR5da99WnBykt1rZDiYck?usp=sharing).

## How to reproduce 
Experiments are reproduced on **one NVIDIA V100**. The random seed is fixed to a value of **42** in two stages.

**NOTE**: Although the code uses `torch.distributed`, running with multiple GPUs has not been tested.

### Stage One: Pretraining 

```sh
python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=xxxx \
mse_tracking_pretrain.py \
--output_dir output/scanpath_prediction_population_pretrain \
```

The model will be saved in `output/scanpath_prediction_population_pretrain`. Before running it, please change your data path in `configs/Pretrain_tracking.yaml`.

>`train_file` is the root path to store csv of fixation points
> 
> `image_root` is the training image root path 
> 
> `eval_image_root` is the testing image root path

### Stage Two: Reinforcement Learning

```sh
python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=xxxx \
mse_tracking.py \
--output_dir output/scanpath_prediction_population_rl \
--checkpoint output/scanpath_prediction_population_pretrain/checkpoint_29.pth \
```

The models will be saved in `output/scanpath_prediction_population_rl`. Before running it, please change your data path in `configs/Tracking.yaml`.

## Predict

```sh
python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=xxxx \
eval_tracking.py \
--output_dir output/scanpath_prediction_population_rl \
--checkpoint output/scanpath_prediction_population_rl/checkpoint_18.pth \
```

The predicted results `predicted_result.csv` will be saved in `output/scanpath_prediction_population_rl`. To reproduce our results, you could use the checkpoint from the 19th epoch.

## Evaluation

We have already preprocessed the testing fixation points in `evaluation/testing_ground_truth.csv`. You can check it and compare it with the raw data.

### DTW, TDE, and Eyenalysis

```sh
python evaluation/eval_xy.py --scanpaths \
--ref_files evaluation/testing_ground_truth.csv \
--pred_files output/scanpath_prediction_population_rl/predicted_result.csv \
```

### DTWD

```sh
python evaluation/eval_xyt.py --scanpaths \
--ref_files evaluation/testing_ground_truth.csv \
--pred_files output/scanpath_prediction_population_rl/predicted_result.csv \
```

### MultiMatch

```sh
cd evaluation/eval_multipath

python eval_tsv.py \
--pred_file your_path/output/scanpath_prediction_population_rl/predicted_result.csv \
--gt_file your_path/evaluation/testing_ground_truth.csv \
```

## Pretrained Model Weights
You can download all the model's weights [here](https://drive.google.com/drive/folders/1nwcDlSDrrI5As68zmvcK9ASaa3kSFzAw?usp=sharing).

## Logging Files
You can now check the two logging files in `running_logs`

## Effects of batch size on model performance
The following experiments are run on one **NVIDIA V100**.

| Batch Size  | Seed   |   DTW  |   TDE  |   Eye   |
|-------------|--------|--------|--------|---------|
| 8           | 42     | 4.2347 | 0.1218 | 0.0370  | 
| 16          | 42     | 4.2624 | 0.1220 | 0.0327  |  
| 32          | 42     | 4.1607 | 0.1244 | 0.0371  |  
| 64          | 42     | 4.1711 | 0.1216 | 0.0361  |
| 64          | 80     | 4.0689 | 0.1222 | 0.0357  |  
