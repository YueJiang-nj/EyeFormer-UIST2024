from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import csv


def extract_scanpaths(csvfile):
    res = []
    cur = None
    obj = None

    with open(csvfile) as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            # print(row)

            # We have a unique combination of image and username per experiment trial.
            uniq = row['image'] + row['username']

            if uniq != cur:
                if 'obj' in locals() and cur is not None:
                    # Store entry.
                    res.append(obj)

                # Init new entry.
                obj = {'image': row['image'], 'width': int(row['width']), 'height': int(row['height']),
                       "username": row['username'], 'scanpath': []}

            # Normalize the scanpath
            x, y = float(row['x']), float(row['y']) #######
            # x, y = float(row['x']) / int(row['width']), float(row['y']) / int(row['height'])  #######
            obj['scanpath'].append([x, y])

            # Flag already seen image.
            cur = uniq

    # Flush last entry.
    if len(obj['scanpath']) > 0:
        res.append(obj)

    return res


def collect_info(res, dir, method):
    images = []
    users = []
    xs_list = []
    ys_list = []
    methods = []

    for r in res:
        # if r["username"] not in ["kh037", "kh012", "kh030", "kh004", "kh043", "kh044", "kh036", "kh062", "kh056"]:
        #     continue
        if method == "Transformer":
            image_path = os.path.join(dir, r["image"])
            if image_path not in images:
                images.append(image_path)
            else:
                continue
        else:
            images.append(os.path.join(dir, r["image"]))
        users.append(r["username"])
        scan_path = r["scanpath"]
        xs = [p[0] for p in scan_path]
        ys = [p[1] for p in scan_path]
        xs_list.append(xs)
        ys_list.append(ys)
        methods.append(method)

    return images, users, xs_list, ys_list, methods


data_dir = "/l/data/Eye-tracking/full/dataset/test"
# blocks = ["block %s" % i for i in range(53, 56)]
# image_list = []
# for block in blocks:
#     image_files = os.listdir(os.path.join(data_dir, block))
#     image_files = [os.path.join(data_dir, block, f) for f in image_files]
#     image_list += image_files


gt_file = "testing_ground_truth.csv"
# pred_file = "../output/eye_tracking_user_transformer/predicted_result.csv"
pred_file = "../output/eye_tracking_general_duration_transformer_saliency_discounted_dtw_rl/predicted_result.csv"
# pred_file = "/l/eye_tracking/personalized_tracking_user_transformer_few_shot/output/image_unique_tracking/predicted_result.csv"

gt_res = extract_scanpaths(gt_file)
pred_res = extract_scanpaths(pred_file)

g_images, g_users, g_xs_list, g_ys_list, g_methods = collect_info(gt_res, data_dir, "Ground_Truth")
p_images, p_users, p_xs_list, p_ys_list, p_methods = collect_info(pred_res, data_dir, "Transformer")

images = g_images + p_images
users = g_users + p_users
xs_list = g_xs_list + p_xs_list
ys_list = g_ys_list + p_ys_list
methods = g_methods + p_methods

colors = [[100, 180, 255, 255],
          [50, 50, 50, 255]]

cm = LinearSegmentedColormap.from_list('', np.array(colors) / 255, 256)



output_dir = './visualization_transformer_saliency_discounted_dtw_rl_2d'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


for aaa in range(len(images)):
    xs = xs_list[aaa]
    ys = ys_list[aaa]
    img = Image.open(images[aaa]).convert("RGB")
    img.putalpha(int(255/2))
    img = np.array(img)

    width = img.shape[1]
    height = img.shape[0]
    w = 1
    h = 1
    plt.gray()
    plt.axis('off')
    ax = plt.imshow(img)
    cmap = (cm(np.linspace(0, 1, 2 * len(xs) - 1)) * 255).astype(np.uint8)

    for i in range(len(xs)):
        if i > 0:
            ax.axes.arrow(
                xs[i - 1]*w,
                ys[i - 1]*h,
                xs[i]*w - xs[i - 1]*w,
                ys[i]*h - ys[i - 1]*h,
                width=min(width, height) / 300,
                color=cmap[i * 2 - 1] / 255.,
                alpha=1,
            )
    for i in range(len(xs)):
        if i == 0:
            edgecolor = 'red'
        else:
            edgecolor = "black"
        cir_rad = min(w, h) / 40
        circle = plt.Circle(
        (xs[i]*w, ys[i]*h),
        radius=min(width, height) / 35,
        edgecolor=edgecolor,
        facecolor=cmap[i * 2] / 255.,
        )
        ax.axes.add_patch(circle)

    # render scanpath on the image
    imagename = os.path.basename(images[aaa]).split(".")[0]
    ax.figure.savefig(output_dir + "/" + '{}_{}_{}.png'.format(imagename, users[aaa], methods[aaa]), dpi=120, bbox_inches="tight")
    plt.close(ax.figure)
