import numpy as np
from scipy.optimize import linear_sum_assignment

from calibration import get_transformation_matrix
from bird_eye_view import get_map_matrix, make_bev_image, image_to_bev_coord
from kalmanfilter import KalmanFilter2D

import pickle
import matplotlib.pyplot as plt

class HungarianMatcher:
    def __init__(self, init_points, threshold):
        self.direction = np.array([0.0, 0.0], dtype=np.float64)

        self.prev_points = init_points

        self.threshold = threshold

    def match_points(self, detected_points):
        if detected_points.ndim != 2:
            return np.array([0.0, 0.0], dtype=np.float64)
        
        num_prev = self.prev_points.shape[0]
        num_detect = detected_points.shape[0]
        cost_matrix = np.zeros((num_prev, num_detect))

        for i in range(num_prev):
            for j in range(num_detect):
                cost_matrix[i, j] = np.linalg.norm(self.prev_points[i] - detected_points[j])

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        diff_vectors = []
        for i, j in zip(row_indices, col_indices):
            distance = cost_matrix[i, j]
            if distance < self.threshold:
                diff_vectors.append(detected_points[j] - self.prev_points[i])

        if len(diff_vectors) > 0:
            diff_vectors = np.array(diff_vectors)

            scales = np.linalg.norm(diff_vectors, axis=1)

            q1 = np.percentile(scales, 25)
            q3 = np.percentile(scales, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            filtered = diff_vectors[(scales >= lower) & (scales <= upper), :]

            avg_vector = np.mean(filtered, axis=0)

            return -avg_vector

            

        return np.array([0.0, 0.0], dtype=np.float64)


if __name__ == "__main__":
    image_shape = np.array([3840, 2160])

    axis_rotation_degree = (90, -90, 0)
    translation = (0, 0, 0)
    rotation_degree = (0, 0, 0)
    fov_degree = 73.7
    camera_height = 0.67

    axis_rotation_radian = np.deg2rad(axis_rotation_degree)
    rotation_radian = np.deg2rad(rotation_degree)
    fov_radian = np.deg2rad(fov_degree)
    m_transformation = get_transformation_matrix(image_shape, fov_radian, axis_rotation_radian, translation, rotation_radian).T

    x_range = (1.5, 12)
    y_range = (-3, 3)
    bev_pixel_interval = (0.015, 0.015)
    
    with open("dataset/bottom_points.pkl", "rb") as f:
        cones_list = pickle.load(f)
    cones_list = [np.array(cones, dtype=np.float64) for cones in cones_list]

    world_points = []
    for cone in cones_list[0]:
        world_point, bev_point = image_to_bev_coord(cone // 4, m_transformation, x_range, y_range, bev_pixel_interval, camera_height)
        world_points.append(world_point)
    world_points = np.array(world_points)
        
    hm = HungarianMatcher(world_points, threshold=5.0)

    dt = 1 / 19.92

    init_state = np.array([0.0, 0.0, 0.0, 0.0])[:, np.newaxis]
    init_cov = np.diag([1.0, 1.0, 1.0, 1.0])
    process_var = 0.1
    measure_var = 0.1

    kf = KalmanFilter2D(dt, init_state, init_cov, process_var, measure_var)

    est_path = []
    for cones in cones_list[1:]:
        world_points = []
        for cone in cones:
            world_point, bev_point = image_to_bev_coord(cone // 4, m_transformation, x_range, y_range, bev_pixel_interval, camera_height)
            world_points.append(world_point)
        world_points = np.array(world_points)

        avg_vector = hm.match_points(world_points)
        hm.prev_points = world_points

        print(avg_vector)

        z = avg_vector[:, np.newaxis] / dt
        kf.predict()
        kf.update(z)

        est_x, est_y, est_vx, est_vy = kf.x.flatten()
        est_path.append((est_x, est_y))
        # print(f"x: {est_x:.3f}, y: {est_y:.3f}")

    est_path = np.array(est_path)

    plt.figure(figsize=(10, 10))
    plt.scatter(est_path[:, 0], est_path[:, 1])
    plt.plot(est_path[:, 0], est_path[:, 1])
    plt.axis("equal")
    plt.show()