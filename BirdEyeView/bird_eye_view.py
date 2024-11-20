import numpy as np
import cv2
import json

def convert_camera_params_to_fov(fx: float, fy: float, image_width: int, image_height: int) -> float:
    """
    표준 카메라 파라미터를 FOV로 변환합니다.
    Args:
        fx (float): X축 초점 거리 (픽셀)
        fy (float): Y축 초점 거리 (픽셀)
        image_width (int): 이미지 너비
        image_height (int): 이미지 높이
    Returns:
        float: 대각선 FOV (라디안)
    """
    # Normalize focal length
    aspect = image_width / image_height
    normalized_fx = fx / (image_width) * aspect  # normalized by half width
    normalized_fy = fy / (image_height) # normalized by half height
    
    # Average normalized focal length
    normalized_f = (normalized_fx + normalized_fy) / 2
    
    # Calculate FOV
    fov = 2 * np.arctan(np.sqrt(1 + aspect**2) / (2 * normalized_f))
    
    return fov

image_path = "./road.png"
image = cv2.imread(image_path)
image_shape = (image.shape[1], image.shape[0])

with open('camera.json', 'r') as f:
    camera_params = json.load(f)

fx = camera_params["fx"]
fy = camera_params["fy"]
aspect = image_shape[0] / image_shape[1]

aspect = (fy / image_shape[1]) / (fx / image_shape[0])
print(aspect)

fov = convert_camera_params_to_fov(fx, fy, image_shape[0], image_shape[1])
print(f"fov: {np.rad2deg(fov)}")

# print(f"focal_length x: {focal_length_x}")
# print(f"focal_length y: {focal_length_y}")