import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

def get_angle_from_object_2d(cam_fov, obj_pos):
    angle1 = obj_pos - 0.5
    tan_theta0 = abs(angle1) / 0.5 * np.tan(cam_fov / 2)
    theta0 = np.arctan(tan_theta0)
    if angle1 < 0:
        theta0 = np.pi / 2 + theta0
    else:
        theta0 = np.pi / 2 - theta0

    return theta0

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



cam1_pos = np.array([-4, 0])
cam2_pos = np.array([4, 0])
object_pos = np.array([0, 5])

fov1 = np.deg2rad(90)
fov2 = np.deg2rad(100)

obj1_pos = (object_pos[1] * np.tan(fov1 / 2) + object_pos[0] - cam1_pos[0]) / (object_pos[1] * np.tan(fov1 / 2) * 2)
obj2_pos = (object_pos[1] * np.tan(fov2 / 2) + object_pos[0] - cam2_pos[0]) / (object_pos[1] * np.tan(fov2 / 2) * 2)


estimated_pos = triangulation2d_test(cam1_pos, cam2_pos, obj1_pos, obj2_pos, fov1, fov2)




plt.figure(figsize=(10, 6))
plt.xlim(-2, 2)
plt.ylim(-1, 4)
plt.grid(True)
plt.axis("equal")

plt.plot(*cam1_pos, 'ro', label='Camera 1')
plt.plot(*cam2_pos, 'go', label='Camera 2')
plt.plot(*object_pos, 'bo', label='Object')
plt.plot(*estimated_pos, 'yo', label='Estimated Object')

def plot_camera_fov(camera_pos, fov, color, length: float = 5):
    rotation = np.pi / 2

    angle_left = rotation + fov / 2
    angle_right = rotation - fov / 2

    fov_pos_left = camera_pos + np.array([np.cos(angle_left), np.sin(angle_left)]) * length
    fov_pos_right = camera_pos + np.array([np.cos(angle_right), np.sin(angle_right)]) * length

    plt.plot([camera_pos[0], fov_pos_left[0]], [camera_pos[1], fov_pos_left[1]], f'{color}--', alpha=0.5)
    plt.plot([camera_pos[0], fov_pos_right[0]], [camera_pos[1], fov_pos_right[1]], f'{color}--', alpha=0.5)

    # plt.plot([camera_pos[0], camera_pos[0]], [camera_pos[1], length * np.cos(fov / 2)], f'{color}--', alpha=0.5)

plot_camera_fov(cam1_pos, fov1, 'r')
plot_camera_fov(cam2_pos, fov2, 'g')

plt.title('Camera and Object Positions')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.show()