import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

def get_angle_from_object_2d(cam_fov, obj_pos):
    angle1 = obj_pos - 0.5
    tan_theta = abs(angle1) / 0.5 * np.tan(cam_fov / 2)
    theta = np.arctan(tan_theta)
    if angle1 < 0:
        theta = np.pi / 2 + theta
    else:
        theta = np.pi / 2 - theta

    return theta

def triangulation2d_test(cam1_pos, cam2_pos, obj1_pos, obj2_pos, cam1_fov, cam2_fov):
    if cam1_pos[0] > cam2_pos[0]:
        cam1_pos, cam2_pos = cam2_pos, cam1_pos
        obj1_pos, obj2_pos = obj2_pos, obj1_pos
        cam1_fov, cam2_fov = cam2_fov, cam1_fov

    theta1 = get_angle_from_object_2d(cam1_fov, obj1_pos)
    theta2 = get_angle_from_object_2d(cam2_fov, obj2_pos)

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

    


        


def triangulation2d(*cameraPos_objectPos_fov: Tuple[np.ndarray, np.ndarray, float]) -> np.ndarray:
    if len(cameraPos_objectPos_fov) < 2:
        raise ValueError("args의 길이가 2 이상이여야 합니다.")
    
    for camera_pos, object_pos, fov in cameraPos_objectPos_fov:
        pass

