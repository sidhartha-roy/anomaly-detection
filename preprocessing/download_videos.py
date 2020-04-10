import os
import csv
import wget
import numpy as np

annotation_file_name = 'annotations.csv'
url = "https://anomaly-recordings.s3.us-west-2.amazonaws.com"

os.makedirs('videos', exist_ok=True)

with open(annotation_file_name, mode='r', encoding='utf-8-sig') as csv_read_file:
    csv_reader = csv.DictReader(csv_read_file)

    for line in csv_reader:
        if line['video'] not in (None, "", '[]') and not os.path.exists('videos/' + line['video']):
            try:
                wget.download(url + '/' + line['video'], 'videos')
            except:
                print('Video file not found for row in annotation file' + line['video'])
                exit()