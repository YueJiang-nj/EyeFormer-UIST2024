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
from torch.utils.data.dataloader import default_collate
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_tracking import TrackingTransformer
from models.vit import interpolate_pos_embed
from models.model_rl import get_self_critical_reward

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from eval_tracking import test
from subprocess import PIPE, run


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=5, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_rl', utils.SmoothedValue(window_size=5, fmt='{value:.4f}'))
    metric_logger.add_meter('rewards', utils.SmoothedValue(window_size=5, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 5
    step_size = 10
    warmup_iterations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, saliency_image, width, height, raw_coord, raw_time) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()

        image = image.to(device, non_blocking=True)
        saliency_image = saliency_image.to(device, non_blocking=True)

        width = width.to(device, non_blocking=True)
        height = height.to(device, non_blocking=True)

        model.eval()
        ### This will be used for calculating the baseline
        with torch.no_grad():
            greedy_pred = model(image, greedy=True)
        model.train()

        sample_pred, sample_pred_prob = model(image, greedy=False)

        ### Calculate the rewards
        rewards = get_self_critical_reward(greedy_pred, raw_coord, raw_time, sample_pred, saliency_image.squeeze(1), width, height)
        rewards = torch.from_numpy(rewards).to(device)

        ### Policy Gradient
        rl_output = - sample_pred_prob * rewards.detach()
        loss = rl_output.mean()

        loss.backward()
        optimizer.step()

        metric_logger.update(loss_rl=loss.item())
        metric_logger.update(rewards=rewards.mean().item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def special_collate(batch_data):
    new_batch_data = [data[:-2] for data in batch_data]
    image, saliency_image, width, height = default_collate(new_batch_data)

    coord = [data[-2] for data in batch_data]
    coord_t = [data[-1] for data in batch_data]
    return image, saliency_image, width, height, coord, coord_t


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset ####
    print("Creating dataset")
    datasets = [create_dataset('tracking', config)]
    test_datasets = [create_dataset('eval_tracking', config)]

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)
    else:
        samplers = [None]

    data_loader = create_loader(datasets, samplers, batch_size=[config['batch_size']], num_workers=[4], is_trains=[True],
                                collate_fns=[special_collate])[0]
    test_loader = create_loader(test_datasets, [None], batch_size=[config['batch_size_test']], num_workers=[4],
                                is_trains=[False], collate_fns=[None])[0]

    # tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    tokenizer = None

    #### Model ####
    print("Creating model")
    model = TrackingTransformer(config=config, init_deit=True)

    model = model.to(device)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        msg = model.load_state_dict(state_dict)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)
        if utils.is_main_process():
            with torch.no_grad():
                test(model_without_ddp, test_loader, tokenizer, device, args.output_dir, config)

            eval_stat_cmd = ["python", "evaluation/eval_xy.py", "--scanpaths", "--ref_files", "evaluation/testing_ground_truth.csv",
                             "--pred_files", "%s/predicted_result.csv" % args.output_dir]

            result = run(eval_stat_cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
            print(result)
            result_out = result.stdout.strip()

            out = [e.strip() for e in result_out.split("\n")[:3]]
            dtw_res = out[0].split(":")[1].strip().split(",")[0][1:]
            tde_res = out[1].split(":")[1].strip().split(",")[0][1:]
            eye_res = out[2].split(":")[1].strip().split(",")[0][1:]
            print("Testing,   DTW: %.4f,   TDE: %.4f,   Eye: %.4f" % (float(dtw_res), float(tde_res), float(eye_res)))

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            save_obj = {
                'model': model_without_ddp.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }

            if (epoch + 1) >= max_epoch - 5:
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Tracking.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='output/tracking')
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