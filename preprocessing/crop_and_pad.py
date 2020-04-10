import os
from PIL import Image, ImageOps
from multiprocessing import Pool, cpu_count

original_directory = 'frames/videos'
target_directory = 'frames/cropped'

os.makedirs(target_directory, exist_ok=True)

def crop(image_path):
    crop_coords = (155,0,225,400)
    saved_location = target_directory + '/' + image_path.split('/')[-2] + '/' + image_path.split('/')[-1]
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(crop_coords)

    old_size = cropped_image.size
    delta_w = 400 - old_size[0]
    delta_h = 400 - old_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

    padded_image = ImageOps.expand(cropped_image, padding)
    padded_image.save(saved_location)
    print('Cropped and padded, saving ', saved_location)


for directory in os.listdir(original_directory):
    try:
        sub_directory = target_directory + '/' + os.fsdecode(directory)
        os.makedirs(sub_directory, exist_ok=True)
        source = original_directory + '/' + os.fsdecode(directory)

        image_paths = []
        for filename in os.listdir(source):
            image = os.fsdecode(source) + '/' + os.fsdecode(filename)
            try:
                image_paths.append(image)
            except:
                continue

        cpus = cpu_count()
        with Pool(cpus) as p:
            p.map(crop, image_paths)

    except:
        continue
