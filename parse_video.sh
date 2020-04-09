#! /bin/bash

if [ ! -f $1 ]; then
    echo "$1 does not exist"
fi

mkdir -p frames/$1

ffmpeg -i $1 -s 400x400 frames/$1/thumb%06d.png -hide_banner