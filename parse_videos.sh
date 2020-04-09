#! /usr/bash

# preprocesses every video in videos

for file in ./videos/*.mp4
do
  echo "File: $file"
  bash parse_video.sh "$file"
done