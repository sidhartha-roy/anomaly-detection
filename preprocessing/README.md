## Instructions

Download the videos by following the instructions below. 
Watch a few of the videos to get a sense of the types of defects that show up. For this problem 
we are only looking at center rope--you should ignore the reflections.
If you want to, you can use some data that we have preprocessed and labeled. 
We have labeled the frames as normal and anomaly and have thrown out a set of "partial" frames. We have also cut out 
the background in the mirrors. 
The below setup may or may not be helpful. 
You can use the labeled data provided or use the csv, and work with the videos directly.

Your task is to spend a few hours (around 4) and write a computer program to catch errors. 
At Overview this is core to what we do. This rope is a simplified case of one of the real world problem we face.

We will run your program on videos similar to these.

If you have technical problems or questions feel free to email Russell his email is: russell@overview.ai

## Setting up data

1. Make sure you have ffmpeg installed. On linux: `sudo apt install ffmpeg`. On mac: `brew install ffmpeg`.
2. Install python requirements `pip install -r requirements.txt`
3. Download Videos `python download_videos.py`
4. Parse Videos `bash parse_videos.sh`
5. Crop and pad `python crop_and_pad.py`
6. Split frames into annotated classes `python setup_data.py`
