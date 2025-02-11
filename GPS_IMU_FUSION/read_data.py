import os
import numpy
import cv2

def read_oxts_data(path):
    with open(path, "r") as f:
        content = f.read()

    data = list(map(float, content.split()))
    data = [data[0], data[1], data[2], data[3], data[4], data[5], data[8], data[11], data[12], data[13]]

    return data

def read_oxts_from_directory(dir, timestamp_path):
    paths = [f for f in os.listdir(dir) if f.endswith('.txt')]
    paths.sort()

    with open(timestamp_path, "r") as f:
        timestamps = f.readlines()
        seconds_list = []
        for timestamp in timestamps:
            time_str = timestamp.strip().split()[-1]
            time_parts = time_str.split(':')
            sec_parts = time_parts[-1].split('.')
            total_seconds = float(time_parts[0]) * 3600 + float(time_parts[1]) * 60 + float(sec_parts[0]) + float(sec_parts[1])/1e+9
            seconds_list.append(total_seconds)

    data = []
    for i, path in enumerate(paths):
        data.append((*read_oxts_data(os.path.join(dir, path)), seconds_list[i]))

    return data

def read_images_from_directory(dir):
    paths = [f for f in os.listdir(dir) if f.endswith('.png')]
    paths.sort()

    images = []
    for path in paths:
        img = cv2.imread(os.path.join(dir, path))
        images.append(img)

    return images

if __name__ == "__main__":
    result = read_oxts_from_directory(r"dataset\oxts\data", r"dataset\oxts\timestamps.txt")
    print([r[-1] for r in result])