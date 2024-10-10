import numpy as np

from typing import Tuple

def get_rotation_matrix_2d(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

def get_angle_from_object_2d(rotation, cam_fov, obj_pos):
    angle1 = obj_pos - 0.5
    tan_theta = abs(angle1) / 0.5 * np.tan(cam_fov / 2)
    theta = np.arctan(tan_theta)
    if angle1 < 0:
        theta = rotation + theta
    else:
        theta = rotation - theta

    return theta

def triangulation2d(cam1_pos, cam1_rotation, cam1_fov, cam2_pos, cam2_rotation, cam2_fov, obj1_pos, obj2_pos):
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
                    obj1_pos: np.ndarray, obj2_pos: np.ndarray):
    m_x_rotation1 = get_rotation_matrix_2d(cam1_rotation[0])
    m_x_rotation2 = get_rotation_matrix_2d(cam2_rotation[0])

    obj1_pos_modified = obj1_pos @ m_x_rotation1
    obj2_pos_modified = obj2_pos @ m_x_rotation2

    

    estimated_pos_y1 = triangulation2d(cam1_pos[[0, 1]], cam1_rotation[1], cam1_fov)

    


