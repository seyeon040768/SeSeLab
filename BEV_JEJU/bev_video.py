import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import pickle
from scipy.optimize import linear_sum_assignment

import calibration
from calibration import get_transformation_matrix
from bird_eye_view import get_map_matrix, make_bev_image, image_to_bev_coord


def icp(source_2d, target_2d, threshold):
    """
    2D point clouds registration using ICP (Iterative Closest Point) algorithm.
    
    Args:
        source_2d: (N, 2) array of source points
        target_2d: (N, 2) array of target points
    
    Returns:
        transformation matrix (4x4)
    """
    # Convert 2D points to 3D by adding z=0
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.hstack([source_2d, np.zeros((source_2d.shape[0], 1))]))
    
    target = o3d.geometry.PointCloud() 
    target.points = o3d.utility.Vector3dVector(np.hstack([target_2d, np.zeros((target_2d.shape[0], 1))]))

    # ICP parameters
    trans_init = np.identity(4)  # Initial transformation
    
    # Perform ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,  # Fix: source and target were swapped
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    T_est = reg_p2p.transformation
    # print(T_est)

    source_hom = np.hstack([source_2d, np.zeros((source_2d.shape[0], 1)), np.ones((source_2d.shape[0], 1))])
    aligned_source_hom = (T_est @ source_hom.T).T
    aligned_source_2d = aligned_source_hom[:, :2]
    
    return T_est, aligned_source_2d

def extract_lane(image, valid_mask):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # Apply Canny edge detection
    canny = cv2.Canny(gray, 50, 150)
    valid_indices = np.where(valid_mask)
    rows = np.unique(valid_indices[0])
    
    for row in rows:
        # Get first valid point
        col_first = valid_indices[1][valid_indices[0] == row][0]
        # Get last valid point 
        col_last = valid_indices[1][valid_indices[0] == row][-1]
        
        # Zero out 3 pixels around first and last valid points
        canny[row, max(0, col_first-1):col_first+2] = 0
        canny[row, max(0, col_last-1):col_last+2] = 0

    # _, white_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
    # canny[white_mask <= 0] = 0

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.morphologyEx(canny, cv2.MORPH_DILATE, kernel)
    
    # Find coordinates of edge pixels
    edge_points = np.column_stack(np.where(dilated.T == 255))
    
    return edge_points

def detect_cone(image, valid_mask):
    with open("dataset/bottom_points.pkl", "rb") as f:
        loaded_list = pickle.load(f)

    return loaded_list


def main():
    # cap = cv2.VideoCapture("dataset/video.mp4")
    cap = cv2.VideoCapture("dataset/track_ob2.mp4")
    
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()

    ret, image = cap.read()
    image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))
    if not ret:
        print("Error: Could not read frame")
        exit()

    image_shape = (image.shape[1], image.shape[0])
    print(image_shape)

    camera_matrix = np.array([
        [345.727618, 0.0, 320.0],
        [0.0, 346.002121, 240.0],
        [0.0, 0.0, 1.0]
    ])
    dist_coeffs = np.array([-0.350124, 0.098598, 0, 0.001998, 0.000177])

    axis_rotation_degree = (90, -90, 0)
    translation = (0, 0, 0)
    rotation_degree = (0, 0, 0)
    fov_degree = 72.5
    camera_height = 0.3

    axis_rotation_radian = np.deg2rad(axis_rotation_degree)
    rotation_radian = np.deg2rad(rotation_degree)
    fov_radian = np.deg2rad(fov_degree)

    h, w = image_shape

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    m_intrinsic = np.eye(4)
    m_intrinsic[:3, :3] = new_camera_matrix
    m_extrinsic = calibration.get_extrinsic_matrix(axis_rotation_radian, translation, rotation_radian)
    m_transformation = (m_intrinsic @ m_extrinsic).T

    x_range = (0.32, 1.7)
    y_range = (-3, 3)
    bev_pixel_interval = (0.005, 0.01)

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

    map_matrix = get_map_matrix(m_transformation, x_range, y_range, bev_pixel_interval, camera_height)
    print(map_matrix.shape)

    projected_points = map_matrix.reshape(-1, 2)

    valid_mask = (0 <= projected_points[:, 0]) & (projected_points[:, 0] <= image_shape[0] - 1) & \
                 (0 <= projected_points[:, 1]) & (projected_points[:, 1] <= image_shape[1] - 1)
    valid_points = projected_points[valid_mask]

    cone_bottoms = detect_cone(None, None)
    prev = None
    index = 0
    point = np.array([0.0, 0.0])
    path = []
    while True:
        ret, image = cap.read()
        if not ret:
            print("Error: Could not read frame") 
            break
        image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))

        bev_image, valid_area = make_bev_image(image, map_matrix, valid_points, valid_mask)

        cones = np.array(cone_bottoms[index])
        world_points = []
        
        for cone in cones:
            world_point, bev_point = image_to_bev_coord(cone // 4, m_transformation, x_range, y_range, bev_pixel_interval, camera_height)
            world_points.append(world_point)
            cv2.circle(bev_image, (int(bev_point[0]), int(bev_point[1])), 5, (255, 0, 0), -1)
            cv2.circle(image, (int(cone[0]//4), int(cone[1]//4)), 5, (255, 0, 0), -1)
        world_points = np.array(world_points)

        avg_vector = np.array([0.0, 0.0])
        if isinstance(prev, np.ndarray):
            count = 0

            plt.figure(figsize=(10, 10))
            plt.scatter(prev[:, 0], prev[:, 1])
            plt.scatter(world_points[:, 0], world_points[:, 1])
            
            num_prev = prev.shape[0]
            num_detect = world_points.shape[0]
            cost_matrix = np.zeros((num_prev, num_detect))

            for i in range(num_prev):
                for j in range(num_detect):
                    cost_matrix[i, j] = np.linalg.norm(prev[i] - world_points[j])

            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            for i, j in zip(row_indices, col_indices):
                distance = cost_matrix[i, j]
                # 비용(거리)이 일정 임계값 이하일 때만 매칭을 유효한 것으로 판단
                threshold = 5  # 필요에 따라 임계값 조정 가능
                if distance < threshold:
                    points = np.vstack([prev[i], world_points[j]])
                    avg_vector += world_points[j] - prev[i]
                    count += 1
                    plt.plot(points[:, 0], points[:, 1])

            avg_vector /= count


            plt.show()
        prev = world_points
        point += avg_vector
        path.append(point.copy())
        avg_vector = avg_vector / np.linalg.norm(avg_vector) * 100
        avg_vector = avg_vector[::-1]
        
        center = np.array((bev_image.shape[1] // 2, bev_image.shape[0] // 2), dtype=np.int32)
        # bev_image = cv2.arrowedLine(bev_image, center, center + avg_vector.astype(np.int32), (255, 0, 0), 5)
        # cv2.imshow("Original", image)
        # cv2.imshow("BEV", bev_image)

        index += 1
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    path = np.array(path)
    print(path)
    plt.figure(figsize=(10, 10))
    plt.scatter(path[:, 0], path[:, 1])
    plt.plot(path[:, 0], path[:, 1])
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()
    