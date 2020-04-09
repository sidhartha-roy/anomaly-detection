# This sets up a directory of anomalous and normal frames for training
import os
import numpy as np
import csv

source_directory = "frames/cropped"
target_directory = "data"
normal_directory = "data/normal"
anomaly_directory = "data/anomaly"
annotation_file = "annotations.csv"
good_bad_ignore_skip = [0, 0, 0, 0]

os.makedirs(normal_directory, exist_ok=True)
os.makedirs(anomaly_directory, exist_ok=True)


def startstop_to_frames(regions, value, frame_array):
    for region in regions:
        frame_array[region[0]:region[1] + 1] = value
        print('region found for ' + str(value))


def load_labels(video_name, length):
    if os.path.exists(annotation_file) == False:
        print('annotation file not downloaded')
        exit()
    with open(annotation_file, mode='r', encoding='utf-8-sig') as csv_read_file:
        csv_reader = csv.DictReader(csv_read_file)
        rowfound = 0
        startstop_anomaly_data = 0
        startstop_ignore_data = 0
        labels = np.zeros(length)

        for line in csv_reader:
            if line['video'] == video_name:
                rowfound = 1

                if line['anomaly_regions'] not in (None, "", '[]'):
                    print('anomaly data found')
                    startstop_anomaly_data = parse_tuple_string(line['anomaly_regions'])
                    startstop_to_frames(startstop_anomaly_data, 1, labels)

                if line['ignore_regions'] not in (None, "", '[]'):
                    startstop_ignore_data = parse_tuple_string(line['ignore_regions'])
                    startstop_to_frames(startstop_ignore_data, 2, labels)

    return (labels)


def parse_tuple_string(s):
    s = s.replace(" ", "")
    return [list(map(int, x.split(","))) for x in s[2:-2].split("],[")]


def process_frames(video_dir, labels):
    for frame_num in range(np.size(labels)):
        frame_str = ("000000" + str(frame_num))[-6:]
        frame_filename = "thumb" + frame_str + ".png"
        frame_path = source_directory + "/" + video_dir + "/" + frame_filename

        if not os.path.exists(frame_path):
            print("Skipping ", frame_path)
            good_bad_ignore_skip[3] += 1
            continue

        source_frame_filename = "../../" + frame_path

        if labels[frame_num] == 1:
            target_frame_filename = anomaly_directory + "/" + video_dir + "-" + frame_filename
            os.symlink(source_frame_filename, target_frame_filename)
            good_bad_ignore_skip[1] += 1

        elif labels[frame_num] == 2:
            print("Throw Out: " + frame_str)
            good_bad_ignore_skip[2] += 1

        elif labels[frame_num] == 0:
            target_frame_filename = normal_directory + "/" + video_dir + "-" + frame_filename
            os.symlink(source_frame_filename, target_frame_filename)
            good_bad_ignore_skip[0] += 1


for video_dir in os.listdir(source_directory):
    try:
        print(video_dir)
        video_dir_path = os.path.join(source_directory, video_dir)
        length = len(os.listdir(video_dir_path))
        print(length)
        labels = load_labels(video_dir, length)
        print(labels)
        process_frames(video_dir, labels)
    except:
        continue
print(good_bad_ignore_skip)