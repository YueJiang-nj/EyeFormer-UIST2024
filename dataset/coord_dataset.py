import json
import os

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import truncate_coord, truncate_time

import csv
from collections import defaultdict
import numpy as np
import copy


def process_data(data, time_data, bpogv_data):
    ### Time is accumulated. The difference is the duration.
    def process_time(item):
        new_item = [0.] + copy.deepcopy(item[:-1])
        time_item = [e1 - e2 for e1, e2 in zip(item, new_item)]
        return time_item

    new_data = defaultdict(dict)

    assert len(data) == len(time_data) == len(bpogv_data)
    keys = list(data.keys())
    for k in keys:
        xy_item = data[k]
        time_item = time_data[k]
        time_item = process_time(time_item)
        v_item = bpogv_data[k]

        new_data[k]["xy"] = []
        new_data[k]["t"] = []

        for xy, time, v in zip(xy_item, time_item, v_item):
            ### Filter out bpogv=0
            if v == 0:
                continue

            ### Filter out center points
            if xy[0] == 0.5 and xy[1] == 0.5:
                continue

            ### Filter out points out of the image
            if xy[0] < 0 or xy[1] < 0 or xy[0] > 1 or xy[1] > 1:
                continue

            new_data[k]["xy"].append(xy)
            new_data[k]["t"].append(time)

        ### Filter out data having no points
        if len(new_data[k]["xy"]) == 0:
            new_data.pop(k)

    return new_data


def read_csv(csv_file, usr_id, train=True):
    filename = os.path.basename(csv_file)
    blk = filename.split("_")[0]
    blk = " ".join(["block", str(int(blk))])

    data = defaultdict(list)
    time_data = defaultdict(list)
    bpogv_data = defaultdict(list)

    with open(csv_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile)

        for idx, row in enumerate(csvreader):
            if idx == 0:
                continue

            media_name = os.path.join(blk, row[1])
            bpogx = float(row[11])
            bpogy = float(row[12])
            bpogv = int(row[13])
            time = float(row[3])

            data[media_name].append([bpogx, bpogy])
            time_data[media_name].append(time)
            bpogv_data[media_name].append(bpogv)

    processed_data = process_data(data, time_data, bpogv_data)

    new_data = []
    for k, v in processed_data.items():
        if train:
            if len(v["xy"]) < 10:
                continue
        ### Add the starting points for coordinates and duration
        append_v = [[0.5, 0.5]] + v["xy"]
        append_t = [0.] + v["t"]
        item = {"image_name": k, "user_name": usr_id, "coord": append_v, "time": append_t}
        new_data.append(item)

    return new_data


def load_ann_file(ann_path, blk_names, train=True):
    data = []

    for usr_dir in os.listdir(ann_path):
        cur_dir = os.path.join(ann_path, usr_dir)
        cur_files = [os.path.join(cur_dir, d) for d in  os.listdir(cur_dir)]
        res_files = []

        for file in cur_files:
            if not file.endswith(".csv"):
                continue

            blk_n = os.path.basename(file).split("_")[0]
            if blk_n not in blk_names:
                continue

            res_files.append(file)

        for res_file in res_files:
            data += read_csv(res_file, usr_dir, train=train)

    return data


def reconstruct_data(data):
    new_data = []
    all_items = defaultdict(dict)
    for e in data:
        image_name = e["image_name"]
        coord = e["coord"]
        time = e["time"]
        if image_name not in all_items:
            all_items[image_name] = {"image_name": image_name, "coord":[], "time": []}
        all_items[image_name]["coord"].append(coord[1:])
        all_items[image_name]["time"].append(time[1:])

    for value in all_items.values():
        new_data.append(value)
    return new_data


class tracking_dataset_pretrain(Dataset):
    def __init__(self, ann_file, image_root, transform, max_words=15):

        image_dir = [os.path.join(image_root, r) for r in os.listdir(image_root)]
        blk_names = [d.split(" ")[-1] for d in image_dir]
        blk_names = ["0"*(2-len(bn))+bn for bn in blk_names]
        self.blk_names = blk_names

        data = load_ann_file(ann_file, blk_names, train=True)

        ### In population level prediction, user2id is not used
        user2id = json.load(open("user2id.json", "r"))

        self.data = data
        self.transform = transform

        ### Max predict steps, i.e., the number of fixation points to predict
        self.max_words = max_words
        self.image_root = image_root
        self.user2id = user2id
        print("Model will generate %s points" % max_words)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        ann = self.data[index]
        user_id = self.user2id[ann["user_name"]]

        image_name = os.path.join(self.image_root, ann["image_name"])
        coord = ann["coord"]
        time = ann["time"]
        assert len(coord) == len(time)

        trunc_coord = truncate_coord(coord, self.max_words)
        trunc_time = truncate_time(time, self.max_words)
        trunc_time = np.array(trunc_time).astype(np.float32)

        trunc_coord_x = np.array([c[0] for c in trunc_coord]).astype(np.float32)
        trunc_coord_y = np.array([c[1] for c in trunc_coord]).astype(np.float32)

        trunc_mask = np.zeros(self.max_words).astype(np.int)

        if len(coord) <= self.max_words:
            trunc_mask[:len(coord)] = 1
        else:
            trunc_mask[:] = 1

        image = Image.open(image_name).convert('RGB')

        width = image.width
        height = image.height

        image = self.transform(image)

        ### Pre-set a radius value of 120
        scalar = min(1920 / width, 1200 / height)
        radius = 120 / scalar
        v_x = radius / width
        v_y = radius / height

        ### Pre-set the duration variance value of 0.1
        v_t = 0.1
        v_xyt = np.array([v_x, v_y, v_t])

        return image, trunc_coord_x, trunc_coord_y, trunc_mask, user_id, trunc_time, v_xyt


class tracking_dataset(Dataset):
    def __init__(self, ann_file, image_root, transform, saliency_transform, max_words=15):

        image_dir = [os.path.join(image_root, r) for r in os.listdir(image_root)]
        blk_names = [d.split(" ")[-1] for d in image_dir]
        blk_names = ["0"*(2-len(bn))+bn for bn in blk_names]
        self.blk_names = blk_names

        data = load_ann_file(ann_file, blk_names, train=True)
        data = reconstruct_data(data)
        user2id = json.load(open("user2id.json", "r"))

        self.saliency_dir = os.path.join(os.path.dirname(ann_file), "saliency_maps")

        self.data = data
        self.transform = transform
        self.saliency_transform = saliency_transform
        self.max_words = max_words
        self.image_root = image_root
        self.user2id = user2id


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        ann = self.data[index]

        image_name = os.path.join(self.image_root, ann["image_name"])
        coord = ann["coord"]
        time = ann["time"]
        assert len(coord) == len(time)

        image = Image.open(image_name).convert('RGB')
        width = image.width
        height = image.height
        image = self.transform(image)

        saliency_image_name = os.path.join(self.saliency_dir, ann["image_name"].replace(" ", "_"))
        saliency_image = Image.open(saliency_image_name).convert("L")
        saliency_image = self.saliency_transform(saliency_image)

        return image, saliency_image, width, height, coord, time


### I think this function is not used
def clean_data(data):
    new_data = []
    visited_image = []
    for d in data:
        image_name = d["image_name"]
        if image_name in visited_image:
            continue
        else:
            visited_image.append(image_name)
            new_data.append(d)

    return new_data


class tracking_dataset_eval(Dataset):
    def __init__(self, ann_file, image_root, transform, max_words=15):

        image_dir = [os.path.join(image_root, r) for r in os.listdir(image_root)]
        blk_names = [d.split(" ")[-1] for d in image_dir]
        blk_names = ["0"*(2-len(bn))+bn for bn in blk_names]
        self.blk_names = blk_names

        data = load_ann_file(ann_file, blk_names, train=False)

        user2id = json.load(open("user2id.json", "r"))
        id2user = {v:k for k, v in user2id.items()}

        self.data = data
        self.transform = transform
        self.max_words = max_words
        self.image_root = image_root
        self.user2id = user2id
        self.id2user = id2user


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        ann = self.data[index]
        user_id = self.user2id[ann["user_name"]]

        image_name = os.path.join(self.image_root, ann["image_name"])

        image = Image.open(image_name).convert('RGB')
        width = image.size[0]
        height = image.size[1]

        image = self.transform(image)

        return image, ann["image_name"], width, height, user_id


