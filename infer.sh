python3 -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=6006 \
inference.py \
--output_dir ./output/ueyes \
--checkpoint ./weights/checkpoint_19.pth \