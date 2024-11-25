import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import os

import calibration

def read_lidar_points(file_path: str) -> np.ndarray:
    """
    txt 파일에서 라이다 포인트를 읽어 nx3 형태의 넘파이 배열로 반환합니다.
    Args:
        file_path (str): 라이다 포인트가 저장된 txt 파일 경로
    Returns:
        np.ndarray: nx3 형태의 라이다 포인트 배열 [x, y, z]
    """
    assert os.path.exists(file_path), f"File does not exist: {file_path}"

    points = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y, z = map(float, line.strip().split())
            points.append([x, y, z])
    
    return np.array(points)

def show_open3d_pcd(pcd, show_origin=True, origin_size=0.1, show_grid=True):
    cloud = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector

    if isinstance(pcd, type(cloud)):
        pass
    elif isinstance(pcd, np.ndarray):
        cloud.points = v3d(pcd)

    coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=origin_size, origin=np.array([0.0, 0.0, 0.0]))

    # set front, lookat, up, zoom to change initial view
    o3d.visualization.draw_geometries([cloud, coord])

if __name__ == "__main__":
    lidar_points = read_lidar_points("0.txt")
    image = cv2.imread("0.png")
    image_shape = (image.shape[1], image.shape[0])

    axis_rotation_degree = (90, -90, 0)
    translation = (0, 0.59, 1.68)
    rotation_degree = (1, 0, 0)
    fov_degree = 30


    m_transformation = calibration.get_transformation_matrix(image_shape, np.deg2rad(fov_degree), np.deg2rad(axis_rotation_degree), translation, np.deg2rad(rotation_degree)).T

    projected_points, valid_indices = calibration.project_points(lidar_points, m_transformation, image_shape)

    
    # show_open3d_pcd(lidar_points[valid_indices])

    for point in projected_points:
        cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("projected_points.png", image)


