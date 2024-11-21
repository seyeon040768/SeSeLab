import numpy as np
import cv2
import matplotlib.pyplot as plt
import calibration
import time
import cProfile
import pstats
from pstats import SortKey

def get_map_matrix(m_transformation: np.ndarray, x_range: tuple[float, float], y_range: tuple[float, float], 
                   bev_pixel_interval: tuple[float, float], camera_height: float) -> np.ndarray:
    """
    Bird's Eye View 변환을 위한 매핑 행렬을 생성합니다.
    Args:
        m_transformation (np.ndarray): 4x4 변환 행렬
        x_range (tuple[float, float]): x축 범위 (최소값, 최대값)
        y_range (tuple[float, float]): y축 범위 (최소값, 최대값)
        bev_pixel_interval (tuple[float, float]): BEV 이미지의 픽셀 간격 (x간격, y간격)
        camera_height (float): 카메라 높이
    Returns:
        np.ndarray: BEV 변환을 위한 매핑 행렬 (높이 x 너비 x 2)
    """
    x_samples = np.arange(x_range[0], x_range[1], bev_pixel_interval[0])[::-1]
    y_samples = np.arange(y_range[0], y_range[1], bev_pixel_interval[1])[::-1]
    output_height = len(x_samples)
    output_width = len(y_samples)

    world_points = np.column_stack([np.repeat(x_samples, output_width), np.tile(y_samples, output_height)])
    world_points = np.hstack([world_points, -camera_height * np.ones((world_points.shape[0], 1)), np.ones((world_points.shape[0], 1))])
    projected_points = world_points @ m_transformation
    projected_points = projected_points[:, :2] / projected_points[:, 2:3]
    map_matrix = projected_points.reshape(output_height, output_width, 2)

    return map_matrix

def make_bev_image(image: np.ndarray, map_matrix: np.ndarray) -> np.ndarray:
    """
    매핑 행렬을 사용하여 Bird's Eye View 이미지를 생성합니다.
    Args:
        image (np.ndarray): 원본 이미지 (HxWx3)
        map_matrix (np.ndarray): BEV 변환을 위한 매핑 행렬 (HxWx2)
    Returns:
        np.ndarray: Bird's Eye View로 변환된 이미지
    """
    output_height = map_matrix.shape[0]
    output_width = map_matrix.shape[1]
    projected_points = map_matrix.reshape(-1, 2)

    bev_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    valid_mask = (0 <= projected_points[:, 0]) & (projected_points[:, 0] <= image_shape[0] - 1) & \
                (0 <= projected_points[:, 1]) & (projected_points[:, 1] <= image_shape[1] - 1)
    valid_points = projected_points[valid_mask]

    floor_coords = np.floor(valid_points).astype(np.int32)
    dists = valid_points - floor_coords

    x0, y0 = floor_coords[:, 0], floor_coords[:, 1]
    x1 = x0 + 1
    y1 = y0 + 1

    left_top_colors = image[y0, x0]
    right_top_colors = image[y0, x1]
    left_bottom_colors = image[y1, x0]
    right_bottom_colors = image[y1, x1]

    dx = dists[:, 0:1]
    dy = dists[:, 1:2]

    interpolated = (left_top_colors * (1 - dx) * (1 - dy) + 
                right_top_colors * dx * (1 - dy) + 
                left_bottom_colors * (1 - dx) * dy + 
                right_bottom_colors * dx * dy)

    bev_image.reshape(-1, 3)[valid_mask] = interpolated

    return bev_image

if __name__ == "__main__":
    # Create a Profile object
    pr = cProfile.Profile()
    
    image_path = "./um_000012.png"
    image = cv2.imread(image_path)
    image_shape = (image.shape[1], image.shape[0])

    rotation_degree = (90, -90, 0)
    translation = (0.06, -7.631618000000e-02, -2.717806000000e-01)
    fov_degree = 85.7
    camera_height = 1.65
    m_transformation = calibration.get_transformation_matrix(image_shape, np.deg2rad(fov_degree), np.deg2rad(rotation_degree), translation).T

    x_range = (6.3, 50)
    y_range = (-10, 10)
    bev_pixel_interval = (0.05, 0.05)

    # Start profiling
    pr.enable()
    
    map_matrix = get_map_matrix(m_transformation, x_range, y_range, bev_pixel_interval, camera_height)
    bev_image = make_bev_image(image, map_matrix)
    
    pr.disable()
    
    stats = pstats.Stats(pr)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)
    
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB), extent=[y_range[0], y_range[1], x_range[0], x_range[1]])
    plt.title('Bird\'s Eye View')
    plt.xlabel('Lateral distance (m)')
    plt.ylabel('Forward distance (m)')
    plt.xticks([-10, -5, 0, 5, 10])
    plt.yticks([6.3, 10, 15, 20, 25, 30, 35, 40, 45, 50])

    plt.tight_layout()
    plt.savefig('bev_result.png')

    plt.show()

