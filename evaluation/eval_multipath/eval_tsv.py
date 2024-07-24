import csv
import os
from subprocess import PIPE, run
import numpy as np
import argparse


def extract_data(file):

    data = {}
    with open(file, "r") as csvfile:

        reader = csv.reader(csvfile)

        for i, row in enumerate(reader):
            if i == 0:
                continue

            image_name = row[0]
            image_name = os.path.basename(image_name)
            width = row[1]
            height = row[2]
            user = row[3]

            data_k = "%s___%s" % (image_name, user)

            if data_k not in data:
                data[data_k] = {"width": width, "height": height, "value": []}

            new_row = row[4:]
            new_row = [float(r) for r in new_row]
            data[data_k]["value"].append(new_row)

    return data

def save_tsv(data, filename):
    with open(filename, "w") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            writer.writerow(["start_x", "start_y", "duration"])
            for d in data:
                writer.writerow(d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', default='./predicted_result.csv')
    parser.add_argument('--gt_file', default='./testing_ground_truth.csv')
    args = parser.parse_args()

    predicted = extract_data(args.pred_file)
    ground_truth = extract_data(args.gt_file)

    keys = list(predicted.keys())

    stats = []

    for i, k in enumerate(keys):
        assert k in ground_truth

        predicted_item = predicted[k]
        ground_truth_item = ground_truth[k]
        if len(ground_truth_item["value"]) < 3:
            continue

        width = ground_truth_item["width"]
        height = ground_truth_item["height"]

        if not os.path.exists("results"):
            os.mkdir("results")

        save_tsv(predicted_item["value"], "results/%s___pd.tsv" % k)
        save_tsv(ground_truth_item["value"], "results/%s___gt.tsv" % k)

        eval_stat_cmd = ["multimatch", "results/%s___pd.tsv" % k, "results/%s___gt.tsv" % k, "--screensize", width, height]

        result = run(eval_stat_cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        result_out = result.stdout.strip()

        result_values = result_out.split("\n")[1:]
        result_values = [v.split("=")[-1].strip() for v in result_values]
        result_values = [float(v) for v in result_values]

        stats.append(result_values)

    stats = np.array(stats)
    mean_stats = np.mean(stats, 0)
    print(mean_stats.tolist())





