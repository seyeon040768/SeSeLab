import numpy as np
import cv2
import json

image_path = "./road.png"
image = cv2.imread(image_path)
image_shape = (image.shape[1], image.shape[0])

with open('camera.json', 'r') as f:
    camera_params = json.load(f)

fx = camera_params["fx"]
fy = camera_params["fy"]
aspect = image_shape[0] / image_shape[1]

focal_length = fx / (image_shape[0] / 2)
fov = 2 * np.arctan(np.sqrt(1 + aspect**2) / focal_length)
fov_degrees = np.rad2deg(fov)

print(f"Image shape: {image_shape}")
print(f"FOV: {fov_degrees:.2f} degrees")