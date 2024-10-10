import numpy as np
import matplotlib.pyplot as plt
import shutil

from triangulation import triangulation2d

import numpy as np

def get_camera_pos2d(camera_pos: np.ndarray, rotation: float, fov: float, object_pos: np.ndarray):
    fov_half = fov / 2
    
    v_camera_direction = np.array([np.cos(rotation), np.sin(rotation)])

    v_cam_to_obj = object_pos - camera_pos
    object_distance = np.linalg.norm(v_cam_to_obj)
    object_cam_angle = np.arccos(np.dot(v_camera_direction, v_cam_to_obj) / object_distance)

    m_left_rotate = np.array([[np.cos(fov_half), np.sin(fov_half)], [-np.sin(fov_half), np.cos(fov_half)]])
    m_right_rotate = np.linalg.inv(m_left_rotate)

    v_left_direction = v_camera_direction @ m_left_rotate
    v_right_direction = v_camera_direction @ m_right_rotate

    fov_length = (object_distance * np.cos(object_cam_angle)) / np.cos(fov_half)

    v_left = v_left_direction * fov_length
    v_right = v_right_direction * fov_length

    v_left_to_obj = v_cam_to_obj - v_left
    v_left_to_right = v_right - v_left

    pos_of_view = np.linalg.norm(v_left_to_obj) / np.linalg.norm(v_left_to_right)
    if (np.dot(v_left_to_obj, v_left_to_right) < 0):
        pos_of_view = -pos_of_view

    return pos_of_view

def plot_camera_fov(camera_pos, rotation, fov, color, length: float = 5):
    angle_left = rotation + fov / 2
    angle_right = angle_left - fov

    fov_pos_left = camera_pos + np.array([np.cos(angle_left), np.sin(angle_left)]) * length
    fov_pos_right = camera_pos + np.array([np.cos(angle_right), np.sin(angle_right)]) * length

    plt.plot([camera_pos[0], fov_pos_left[0]], [camera_pos[1], fov_pos_left[1]], f'{color}--', alpha=0.5)
    plt.plot([camera_pos[0], fov_pos_right[0]], [camera_pos[1], fov_pos_right[1]], f'{color}--', alpha=0.5)

cam1_pos = np.array([-4, 0])
cam1_rotation = np.deg2rad(95)
cam1_fov = np.deg2rad(100)
cam1_fov_half = cam1_fov / 2

cam2_pos = np.array([4, 0])
cam2_rotation = np.deg2rad(85)
cam2_fov = np.deg2rad(100)
cam2_fov_half = cam2_fov / 2

object_pos = np.array([0, 5])

obj1_pos = get_camera_pos2d(cam1_pos, cam1_rotation, cam1_fov, object_pos)
obj2_pos = get_camera_pos2d(cam2_pos, cam2_rotation, cam2_fov, object_pos)


estimated_pos = triangulation2d(cam1_pos, cam1_rotation, cam1_fov, cam2_pos, cam2_rotation, cam2_fov, obj1_pos, obj2_pos)


print(f"camera1\t\tpos\t\trotation\tfov(degree)\tobject\n\t\t({cam1_pos[0]:.2f}, {cam1_pos[1]:.2f})\t{np.rad2deg(cam1_rotation):.2f}\t\t{np.rad2deg(cam1_fov):.2f}\t\t{obj1_pos:.2f}")
print(f"camera2\t\tpos\t\trotation\tfov(degree)\tobject\n\t\t({cam2_pos[0]:.2f}, {cam2_pos[1]:.2f})\t{np.rad2deg(cam2_rotation):.2f}\t\t{np.rad2deg(cam2_fov):.2f}\t\t{obj2_pos:.2f}")
print("-" * shutil.get_terminal_size().columns)
print(f"object_pos\testimated_pos\n({object_pos[0]:.2f}, {object_pos[1]:.2f})\t({estimated_pos[0]:.2f}, {estimated_pos[1]:.2f})")


plt.figure(figsize=(10, 6))
plt.xlim(-2, 2)
plt.ylim(-1, 4)
plt.grid(True)
plt.axis("equal")

plt.plot(*cam1_pos, 'ro', label='Camera 1')
plt.plot(*cam2_pos, 'go', label='Camera 2')
plt.plot(*object_pos, 'bo', label='Object')
plt.plot(*estimated_pos, 'yo', label='Estimated Object')

plot_camera_fov(cam1_pos, cam1_rotation, cam1_fov, 'r')
plot_camera_fov(cam2_pos, cam2_rotation, cam2_fov, 'g')

plt.title('Camera and Object Positions')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.show()