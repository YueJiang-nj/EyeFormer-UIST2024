import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.coord_dataset import tracking_dataset_pretrain, tracking_dataset, tracking_dataset_eval

from dataset.randaugment import RandomAugment


def create_dataset(dataset, config):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    tracking_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    saliency_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])

    if dataset == 'pretrain':
        dataset = tracking_dataset_pretrain(config['train_file'], config['image_root'], tracking_transform, max_words=config["max_words"])
        return dataset

    elif dataset == 'tracking':
        dataset = tracking_dataset(config['train_file'], config['image_root'], tracking_transform, saliency_transform, max_words=config["max_words"])
        return dataset

    elif dataset == 'eval_tracking':
        dataset = tracking_dataset_eval(config['train_file'], config['eval_image_root'], tracking_transform, max_words=config["max_words"])
        return dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders