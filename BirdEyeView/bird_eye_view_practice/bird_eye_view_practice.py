import numpy as np
import cv2
import matplotlib.pyplot as plt
import calibration
from copy import deepcopy

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
    image_shape = (image.shape[1], image.shape[0])
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

def get_world_points(points: np.ndarray, camera_height: float, image_shape: tuple[int, int], fov: float) -> np.ndarray:
    w = points[:, 0] / (image_shape[1] / 2)
    h = points[:, 1] / (image_shape[1] / 2)

    camera_height = -camera_height

    aspect = image_shape[0] / image_shape[1]

    focal_length = np.sqrt(1 + aspect**2) / np.tan(fov / 2)

    u0 = aspect
    v0 = 1

    eps = 1e-12

    x = (focal_length * camera_height) / (v0 - h - eps)
    y = camera_height * (u0 - w) / (v0 - h - eps)
    z = np.full_like(x, camera_height)

    return np.column_stack([x, y, z, np.ones_like(x)])

def draw_grid(image: np.ndarray, m_transformation: np.ndarray, camera_height: float, x_end: float, grid_size: tuple[float, float],
              line_color: tuple[int, int, int]=(255, 255, 255), line_thickness: int=1) -> tuple[np.ndarray, tuple[tuple[float, float], tuple[float, float]]]:
    return_image = deepcopy(image)
    image_shape = (image.shape[1], image.shape[0])

    center_buttom_point = np.array([image_shape[0]//2, image_shape[1]-1])
    left_bottom_point = np.array([0, image_shape[1]-1])
    right_bottom_point = np.array([image_shape[0]-1, image_shape[1]-1])
    border_points = np.vstack([center_buttom_point, left_bottom_point, right_bottom_point])

    world_points = get_world_points(border_points, camera_height, image_shape, np.deg2rad(fov_degree))

    x_start = world_points[0, 0]
    y_range = (world_points[2, 1], world_points[1, 1])

    center_far_point = np.array([x_end, 0, -camera_height, 1])
    left_far_point = np.array([x_end, y_range[1], -camera_height, 1])
    right_far_point = np.array([x_end, y_range[0], -camera_height, 1])
    world_points = np.vstack([world_points, center_far_point, left_far_point, right_far_point])
    
    x_locations = np.arange(x_end, x_start, -grid_size[0])
    x_left_points = np.column_stack([x_locations, np.full_like(x_locations, y_range[0]), np.full_like(x_locations, -camera_height)])
    x_left_projected_points = calibration.project_points(x_left_points, m_transformation, image_shape)[0]
    x_left_projected_points = x_left_projected_points[:, :2].astype(np.int32)

    x_right_points = np.column_stack([x_locations, np.full_like(x_locations, y_range[1]), np.full_like(x_locations, -camera_height)])
    x_right_projected_points = calibration.project_points(x_right_points, m_transformation, image_shape)[0]
    x_right_projected_points = x_right_projected_points[:, :2].astype(np.int32)

    y_locations = np.concatenate([np.arange(0, y_range[0], -grid_size[1])[1:], np.arange(0, y_range[1], grid_size[1])])
    y_close_points = np.column_stack([np.full_like(y_locations, x_start), y_locations, np.full_like(y_locations, -camera_height)])
    y_close_projected_points = calibration.project_points(y_close_points, m_transformation, image_shape)[0]
    y_close_projected_points = y_close_projected_points[:, :2].astype(np.int32)

    y_far_points = np.column_stack([np.full_like(y_locations, x_end), y_locations, np.full_like(y_locations, -camera_height)])
    y_far_projected_points = calibration.project_points(y_far_points, m_transformation, image_shape)[0]
    y_far_projected_points = y_far_projected_points[:, :2].astype(np.int32)

    for x_left, x_right in zip(x_left_projected_points, x_right_projected_points):
        cv2.line(return_image, x_left, x_right, line_color, line_thickness)
    for y_close, y_far in zip(y_close_projected_points, y_far_projected_points):
        cv2.line(return_image, y_close, y_far, line_color, line_thickness)

    return return_image, ((x_start, x_end), y_range)

if __name__ == "__main__":
    image_shape = (6400, 4800)
    example_image = np.zeros((image_shape[1], image_shape[0], 3), dtype=np.uint8)
    example_image[:, :, :] = 255

    rotation_degree = (90, -90, 0)
    translation = (0, 0, 0)
    fov_degree = 90
    camera_height = 1
    m_transformation = calibration.get_transformation_matrix(image_shape, np.deg2rad(fov_degree), np.deg2rad(rotation_degree), translation).T

    grid_image, (x_range, y_range) = draw_grid(example_image, m_transformation, camera_height, 15, (0.5, 0.5), (0, 0, 0), 7)

    map_matrix = get_map_matrix(m_transformation, x_range, (-2, 2), (0.01, 0.01), camera_height)
    bev_image = make_bev_image(grid_image, map_matrix)
    print(bev_image.shape)
        
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
    plt.title('Grid Image(Grid Size: 0.5m)')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB), extent=[y_range[0], y_range[1], x_range[0], x_range[1]])
    plt.title('Bird\'s Eye View')
    plt.xlabel('Lateral distance (m)')
    plt.ylabel('Forward distance (m)')

    plt.tight_layout()
    plt.savefig('bev_result_practice.png')

    plt.show()
