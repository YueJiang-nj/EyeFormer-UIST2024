'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_tracking import TrackingTransformer
from models.vit import interpolate_pos_embed

import utils
from dataset import create_dataset, create_sampler, create_loader
import csv


@torch.no_grad()
def test(model, data_loader, tokenizer, device, output_dir, config):
    # train
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Testing:'

    image_names = []
    results = []
    widths = []
    heights = []
    # user_ids = []
    for i, (image, image_name, width, height) in enumerate(metric_logger.log_every(data_loader, 1, header)):
        image = image.to(device, non_blocking=True)
        # user_id = user_id.to(device, non_blocking=True)

        coord = model.inference(image)
        coord = coord.cpu().numpy().tolist()

        width = width.numpy().tolist()
        height = height.numpy().tolist()

        # user_id = user_id.cpu().numpy().tolist()

        image_names.extend(image_name)
        results.extend(coord)
        widths.extend(width)
        heights.extend(height)
        # user_ids.extend(user_id)

    with open(os.path.join(output_dir, 'predicted_result.csv'), 'w') as wfile:
        writer = csv.writer(wfile)
        writer.writerow(["image", "width", "height", "x", "y", "timestamp"])

        for image, width, height, coord in zip(image_names, widths, heights, results):

            for row in coord:
                x = row[0] * width
                y = row[1] * height
                t = row[2]
                # username = data_loader.dataset.id2user[user_id]
                writer.writerow([image, width, height,
                                x, y, t])

    return


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    datasets = [create_dataset('inference', config)]

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)
    else:
        samplers = [None]

    data_loader = create_loader(datasets, 
                                samplers, 
                                batch_size=[config['batch_size_test']], 
                                num_workers=[32], 
                                is_trains=[False],
                                collate_fns=[None])[0]

    # tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    tokenizer = None

    #### Model ####
    print("Creating model")
    model = TrackingTransformer(config=config, init_deit=False)

    model = model.to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        msg = model.load_state_dict(state_dict)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model

    print("Start testing")
    start_time = time.time()

    test(model, data_loader, tokenizer, device, args.output_dir, config)

    dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Tracking.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='output/tracking_eval')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)