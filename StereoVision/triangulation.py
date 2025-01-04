import numpy as np

from typing import Tuple

def decompose_fov_hv(fov: float, image_shape: Tuple) -> Tuple[float, float]:
    """decompose diagonal fov into horizontal, vertical fov

    Args:
        fov (float): diagonal fov
        image_shape (Tuple): shape of image(2d)

    Returns:
        Tuple[float, float]: horizontal fov, vertical fov
    """
    aspect = image_shape[0] / image_shape[1]
    f = np.sqrt(aspect**2 + 1) / np.tan(fov / 2)
    fov_h = np.arctan2(aspect, f) * 2
    fov_v = np.arctan2(1, f) * 2

    return fov_h, fov_v

def get_rotation_matrix_2d(theta: float) -> np.ndarray:
    """return 2d rotation matrix(2x2)

    Args:
        theta (float): rotation angle

    Returns:
        np.ndarray: 2d rotation matrix
    """
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

def get_angle_from_object_2d(rotation: float, cam_fov: float, obj_pos: float) -> float:
    """get angle between camera and object based on polar axis

    Args:
        rotation (float): rotation of camera
        cam_fov (float): camera field of view
        obj_pos (float): position of object, fov boundary [0, 1]

    Returns:
        float: angle
    """
    displacement_from_center = obj_pos - 0.5
    tan_theta = abs(displacement_from_center) / (0.5 / np.tan(cam_fov / 2)) # disp / cam_to_center
    theta = np.arctan(tan_theta)
    if displacement_from_center < 0:
        theta = rotation + theta
    else:
        theta = rotation - theta

    return theta

def triangulation2d(cam1_pos: np.ndarray, cam1_rotation: float, cam1_fov: float,
                    cam2_pos: np.ndarray, cam2_rotation: float, cam2_fov: float,
                    obj1_pos: float, obj2_pos: float) -> np.ndarray:
    """triangulate from 2d space with 2 cameras

    Args:
        cam1_pos (np.ndarray): position of camera1
        cam1_rotation (float): rotation of camera1(from polar axis)
        cam1_fov (float): field-of-view of camera1
        cam2_pos (np.ndarray): position of camera2
        cam2_rotation (float): rotation of camera2(from polar axis)
        cam2_fov (float): field-of-view of camera2
        obj1_pos (float): position of object, captured by camera1
        obj2_pos (float): position of object, captured by camera2

    Returns:
        np.ndarray: estimated position of object(2d)
    """
    if cam1_pos[0] > cam2_pos[0]:
        cam1_pos, cam2_pos = cam2_pos, cam1_pos
        obj1_pos, obj2_pos = obj2_pos, obj1_pos
        cam1_fov, cam2_fov = cam2_fov, cam1_fov

    theta1 = get_angle_from_object_2d(cam1_rotation, cam1_fov, obj1_pos)
    theta2 = get_angle_from_object_2d(cam2_rotation, cam2_fov, obj2_pos)

    alpha = theta1
    beta = np.pi - theta2
    gamma = np.pi - alpha - beta

    base = cam2_pos[0] - cam1_pos[0]

    distance1 = base * np.sin(beta) / np.sin(gamma)
    distance2 = base * np.sin(alpha) / np.sin(gamma)

    estimated_pos1 = distance1 * np.array([np.cos(theta1), np.sin(theta1)]) + cam1_pos
    estimated_pos2 = distance2 * np.array([np.cos(theta2), np.sin(theta2)]) + cam2_pos

    estimated_pos = np.mean((estimated_pos1, estimated_pos2), axis=0)

    return estimated_pos

def triangulation3d(cam1_pos: np.ndarray, cam1_rotation: np.ndarray, cam1_fov: float, img1_shape: Tuple,
                    cam2_pos: np.ndarray, cam2_rotation: np.ndarray, cam2_fov: float, img2_shape: Tuple,
                    obj1_pos: np.ndarray, obj2_pos: np.ndarray) -> np.ndarray:
    """triangulate from 3d space with 2 cameras

    Args:
        cam1_pos (np.ndarray): position of camera1
        cam1_rotation (np.ndarray): rotation of camera1(x, y, z)
        cam1_fov (float): field-of-view of camera1
        img1_shape (Tuple): (width, height)
        cam2_pos (np.ndarray): position of camera2
        cam2_rotation (np.ndarray): rotation of camera2(x, y, z)
        cam2_fov (float): field-of-view of camera2
        img2_shape (Tuple): (width, height)
        obj1_pos (np.ndarray): position of object, captured by camera1
        obj2_pos (np.ndarray): position of object, captured by camera2

    Returns:
        np.ndarray: estimated position of object(3d)
    """
    m_x_rotation1 = get_rotation_matrix_2d(cam1_rotation[0])
    m_x_rotation2 = get_rotation_matrix_2d(cam2_rotation[0])

    obj1_pos_modified = obj1_pos @ m_x_rotation1
    obj2_pos_modified = obj2_pos @ m_x_rotation2

    cam1_fov_h, cam1_fov_v = decompose_fov_hv(cam1_fov, img1_shape)
    cam2_fov_h, cam2_fov_v = decompose_fov_hv(cam2_fov, img2_shape)

    estimated_pos_xy = triangulation2d(cam1_pos[[0, 1]], cam1_rotation[1], cam1_fov_h, cam2_pos[[0, 1]], cam2_rotation[1], cam2_fov_h, obj1_pos_modified[[0, 1]], obj2_pos_modified[[0, 1]])
    estimated_pos_xz = triangulation2d(cam1_pos[[0, 2]], cam1_rotation[2], cam1_fov_v, cam2_pos[[0, 2]], cam2_rotation[2], cam2_fov_v, obj1_pos_modified[[0, 2]], obj2_pos_modified[[0, 2]])
    estimated_pos = np.array([(estimated_pos_xy[0] + estimated_pos_xz[0]) / 2, estimated_pos_xy[1], estimated_pos_xz[1]])

    return estimated_pos

def compute_distance_map(disparity_map, focal_length, baseline):
    """
    거리 맵을 계산합니다.
    
    Args:
        disparity_map: 시차 맵 (픽셀 단위)
        focal_length: 초점 거리 (픽셀 단위)
        baseline: 스테레오 카메라 간 거리 (미터 단위)
    
    Returns:
        distance_map: 거리 맵 (미터 단위)
    """
    disparity_map = disparity_map.astype(np.float32)
    
    # 유효하지 않은 disparity 처리
    invalid_disparity = (disparity_map <= 0)
    
    # 거리 계산 공식: distance = (focal_length * baseline) / disparity
    distance_map = np.zeros_like(disparity_map)
    distance_map[~invalid_disparity] = (focal_length * baseline) / disparity_map[~invalid_disparity]
    distance_map[invalid_disparity] = np.nan
    
    return distance_map