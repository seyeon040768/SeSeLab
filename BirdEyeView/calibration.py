import numpy as np

def get_rotation_matrix_x(theta: float) -> np.ndarray:
    """
    X축을 기준으로 회전 행렬을 생성합니다.
    Args:
        theta (float): 회전 각도 (라디안)
    Returns:
        np.ndarray: 4x4 회전 행렬
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    m_rotation = np.array([
        [1, 0, 0, 0],
        [0, cos_theta, -sin_theta, 0],
        [0, sin_theta, cos_theta, 0],
        [0, 0, 0, 1]
    ])

    return m_rotation

def get_rotation_matrix_y(theta: float) -> np.ndarray:
    """
    Y축을 기준으로 회전 행렬을 생성합니다.
    Args:
        theta (float): 회전 각도 (라디안)
    Returns:
        np.ndarray: 4x4 회전 행렬
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    m_rotation = np.array([
        [cos_theta, 0, sin_theta, 0],
        [0, 1, 0, 0],
        [-sin_theta, 0, cos_theta, 0],
        [0, 0, 0, 1]
    ])

    return m_rotation

def get_rotation_matrix_z(theta: float) -> np.ndarray:
    """
    Z축을 기준으로 회전 행렬을 생성합니다.
    Args:
        theta (float): 회전 각도 (라디안)
    Returns:
        np.ndarray: 4x4 회전 행렬
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    m_rotation = np.array([
        [cos_theta, -sin_theta, 0, 0],
        [sin_theta, cos_theta, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return m_rotation

def get_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """
    3차원 회전 벡터를 이용하여 3차원 회전 행렬을 생성합니다.
    Args:
        rotation (np.ndarray): [roll, pitch, yaw] 회전 벡터 (라디안)
    Returns:
        np.ndarray: 4x4 회전 행렬
    """
    roll, pitch, yaw = rotation
    m_rotation = get_rotation_matrix_z(yaw) @ get_rotation_matrix_y(pitch) @ get_rotation_matrix_x(roll)
    return m_rotation

def get_translation_matrix(translation: np.ndarray) -> np.ndarray:
    """
    이동 행렬을 생성합니다.
    Args:
        translation (np.ndarray): [x, y, z] 이동 벡터
    Returns:
        np.ndarray: 4x4 이동 행렬
    """
    x, y, z = translation
    m_translation = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])
    return m_translation

def get_extrinsic_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    외부 파라미터 행렬을 생성합니다.
    Args:
        rotation (np.ndarray): [roll, pitch, yaw] 회전 벡터 (라디안)
        translation (np.ndarray): [x, y, z] 이동 벡터
    Returns:
        np.ndarray: 4x4 외부 파라미터 행렬
    """
    m_extrinsic = get_translation_matrix(translation) @ get_rotation_matrix(rotation)
    return m_extrinsic

def get_intrinsic_matrix(fov: float, aspect: float) -> np.ndarray:
    """
    내부 파라미터 행렬을 생성합니다.
    (2*aspect, 2) 크기의 투영 평면에 점을 투영합니다.
    Args:
        fov (float): 대각선 FOV (라디안)
        aspect (float): 투영 평면의 가로세로 비율
    Returns:
        np.ndarray: 4x4 내부 파라미터 행렬
    """
    focal_length = np.sqrt(1 + aspect**2) / np.tan(fov / 2)

    # 카메라에 따라 fx, fy를 사용하는 것이 더 정확한 경우도 있음
    fov_y = 2 * np.arctan(1 / focal_length)
    fov_x = 2 * np.arctan(aspect / focal_length)

    fy = 1 / np.tan(fov_y / 2)
    fx = 1 / np.tan(fov_x / 2)

    u0 = aspect
    v0 = 1

    m_intrinsic = np.array([
        [focal_length, 0, u0, 0],
        [0, focal_length, v0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return m_intrinsic

def get_expand_matrix(image_height: int) -> np.ndarray:
    """
    이미지 확장 행렬을 생성합니다.
    Args:
        image_height (int): 이미지 높이
    Returns:
        np.ndarray: 4x4 확장 행렬
    """
    m_expand = np.array([
        [image_height / 2, 0, 0, 0],
        [0, image_height / 2, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return m_expand

def get_transformation_matrix(image_shape: tuple[int, int], fov: float,
                            rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    3D 좌표를 2D 이미지 좌표로 변환하는 변환 행렬을 생성합니다.
    Args:
        image_shape (tuple[int, int]): 이미지 크기 (너비, 높이)
        fov (float): 시야각 (라디안)
        rotation (np.ndarray): [roll, pitch, yaw] 회전 벡터 (라디안)
        translation (np.ndarray): [x, y, z] 이동 벡터
    Returns:
        np.ndarray: 4x4 변환 행렬
    """
    m_extrinsic = get_extrinsic_matrix(rotation, translation)
    m_intrinsic = get_intrinsic_matrix(fov, image_shape[0] / image_shape[1])
    m_expand = get_expand_matrix(image_shape[1])

    m_transformation = m_expand @ m_intrinsic @ m_extrinsic

    return m_transformation

def project_points(points: np.ndarray, m_transformation: np.ndarray, 
                  image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    3D 점들을 2D 이미지 평면에 투영합니다.
    Args:
        points (np.ndarray): 3D 점들의 배열 (Nx3 또는 Nx4)
        m_transformation (np.ndarray): 4x4 변환 행렬
        image_shape (tuple[int, int]): 이미지 크기 (너비, 높이)
    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - 투영된 2D 점들의 배열
            - 유효한 점들의 인덱스
    """
    points = np.concatenate([points[:, :3], np.ones((points.shape[0], 1))], axis=1)

    points = points @ m_transformation
    points[:, 0] = points[:, 0] / points[:, 2]
    points[:, 1] = points[:, 1] / points[:, 2]

    valid_indices = np.where(
        (points[:, 0] >= 0) & 
        (points[:, 0] < image_shape[0]) & 
        (points[:, 1] >= 0) & 
        (points[:, 1] < image_shape[1]) &
        (points[:, 2] > 0)
    )[0]
    points = points[valid_indices]

    return points, valid_indices

def inverse_project_points(points: np.ndarray, m_transformation: np.ndarray) -> np.ndarray:
    """
    2D 이미지 평면의 점들을 3D 공간으로 역투영합니다.
    Args:
        points (np.ndarray): 2D 점들의 배열 (Nx4, 3번째 열은 깊이값)
        m_transformation (np.ndarray): 4x4 변환 행렬
    Returns:
        np.ndarray: 역투영된 3D 점들의 배열 (Nx4)
    """
    points = np.concatenate([points[:, :3], np.ones((points.shape[0], 1))], axis=1)

    points[:, 0] = points[:, 0] * points[:, 2]
    points[:, 1] = points[:, 1] * points[:, 2]

    points = points @ np.linalg.inv(m_transformation)

    return points

if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    image = cv2.imread("um_000012.png")
    image_shape = (image.shape[1], image.shape[0])

    points = np.array([
        [6.3, 0, -1.65],
        [6.3, 15, -1.65],
        [6.3, -15, -1.65],
        [50.0, 0, -1.65],
        [50.0, 10, -1.65],
        [50.0, -10, -1.65],
    ])

    rotation_degree = (90, -90, 0)
    # translation = (0.06, -0.08, -0.27)
    translation = (0.06, -7.631618000000e-02, -2.717806000000e-01)
    fov_degree = 85.7

    m_transformation = get_transformation_matrix(image_shape, np.deg2rad(fov_degree), np.deg2rad(rotation_degree), translation).T

    projected_points, valid_indices = project_points(points, m_transformation, image_shape)
    
    for point in projected_points:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

