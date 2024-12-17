import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import calibration
import bird_eye_view
import os
from tqdm import tqdm

def process_image(image_path, m_transformation, x_range, y_range, bev_pixel_interval, camera_height, x_roi):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_shape = (image.shape[1], image.shape[0])

    map_matrix = bird_eye_view.get_map_matrix(m_transformation, x_range, y_range, bev_pixel_interval, camera_height)
    bev_image = bird_eye_view.make_bev_image(image, map_matrix)
    hsl_image = cv2.cvtColor(bev_image, cv2.COLOR_RGB2HLS)

    green_channel = bev_image[:, :, 1]
    lightness_channel = hsl_image[:, :, 1]

    blur = cv2.GaussianBlur(lightness_channel, (5, 5), 0)
    edge = cv2.Canny(blur, 30, 100)

    x_roi_mask = np.zeros_like(edge, dtype=np.uint8)
    x_roi_mask = cv2.rectangle(x_roi_mask, (x_roi[0], 0), (x_roi[1], x_roi_mask.shape[0]), (255, 255), -1)
    edge_roi = cv2.bitwise_and(edge, edge, mask=x_roi_mask)
    edge_roi = edge_roi[:, x_roi[0]:x_roi[1]]

    return bev_image, edge_roi

if __name__ == "__main__":
    base_dir = "./TUSimple/test_set/clips/0530"
    folders = sorted(os.listdir(base_dir))
    data_paths = []
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            data_paths.extend([os.path.join(folder, f) for f in sorted(os.listdir(folder_path))])

    image_path = "1.jpg"
    

    axis_rotation = (90, -90, 0)
    translation = (0, 0, 0) # camera 위치 기준
    rotation = (5.16, 0, 0)
    fov_degree = 60
    camera_height = 1.5
    image_shape = (1280, 720)

    x_range = (10, 50)
    y_range = (-20, 20)
    bev_pixel_interval = (0.05, 0.05)

    result_shape = np.ceil(((y_range[1] - y_range[0]) / bev_pixel_interval[1], (x_range[1] - x_range[0]) / bev_pixel_interval[0])).astype(np.int32)

    # x_roi = (290, 464)
    x_roi = (0, result_shape[0]-1)
    m_transformation = calibration.get_transformation_matrix(image_shape, np.deg2rad(fov_degree), np.deg2rad(axis_rotation), translation, np.deg2rad(rotation)).T

    bev_image, edge_roi = process_image(image_path, m_transformation, x_range, y_range, bev_pixel_interval, camera_height, x_roi)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(bev_image)
    plt.subplot(122)
    plt.imshow(edge_roi, cmap='gray')
    plt.show()