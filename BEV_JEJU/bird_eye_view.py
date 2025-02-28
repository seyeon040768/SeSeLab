import numpy as np
import cv2
import matplotlib.pyplot as plt
from calibration import get_transformation_matrix

class BEV:
    def __init__(self, image_shape, m_transformation, x_range, y_range, bev_pixel_interval, camera_height, boundary_margin=2):
        self.origin_image_shape = image_shape
        self.m_transformation = m_transformation
        self.x_range = x_range
        self.y_range = y_range
        self.bev_pixel_interval = bev_pixel_interval
        self.camera_height = camera_height

        self.map_matrix = self.get_map_matrix()
        self.output_height, self.output_width = self.map_matrix.shape[:2]
        self.bev_shape = np.array([self.output_width, self.output_height], dtype=np.int32)

        projected_points = self.map_matrix.reshape(-1, 2)

        self.valid_mask = (0 <= projected_points[:, 0]) & (projected_points[:, 0] <= self.origin_image_shape[0] - 1) & \
                    (0 <= projected_points[:, 1]) & (projected_points[:, 1] <= self.origin_image_shape[1] - 1)
        self.valid_area = self.valid_mask.reshape((self.output_height, self.output_width))
        self.valid_points = projected_points[self.valid_mask]

        rows_valid = np.where(self.valid_area.any(axis=1))[0]

        first_cols = np.argmax(self.valid_area, axis=1)[rows_valid]
        last_cols = self.valid_area.shape[1] - np.argmax(self.valid_area[:, ::-1], axis=1)[rows_valid] - 1

        first_margin = np.stack([first_cols - boundary_margin, first_cols, first_cols + boundary_margin], axis=1)
        last_margin  = np.stack([last_cols - boundary_margin, last_cols, last_cols + boundary_margin], axis=1)

        rows_first = np.tile(rows_valid[:, None], (1, 3))
        rows_last  = np.tile(rows_valid[:, None], (1, 3))

        valid_first = (first_margin >= 0) & (first_margin < self.bev_shape[0])
        valid_last  = (last_margin >= 0) & (last_margin < self.bev_shape[0])

        self.boundary_mask = np.zeros_like(self.valid_area, dtype=bool)
        self.boundary_mask[rows_first[valid_first], first_margin[valid_first]] = True
        self.boundary_mask[rows_last[valid_last], last_margin[valid_last]] = True

        self.valid_area_margined = self.valid_area & ~self.boundary_mask


    def get_map_matrix(self) -> np.ndarray:
        x_samples = np.arange(self.x_range[0], self.x_range[1]+self.bev_pixel_interval[0], self.bev_pixel_interval[0])[::-1]
        y_samples = np.arange(self.y_range[0], self.y_range[1]+self.bev_pixel_interval[1], self.bev_pixel_interval[1])[::-1]
        output_height = len(x_samples)
        output_width = len(y_samples)

        world_points = np.column_stack([np.repeat(x_samples, output_width), np.tile(y_samples, output_height)])
        world_points = np.hstack([world_points, -self.camera_height * np.ones((world_points.shape[0], 1)), np.ones((world_points.shape[0], 1))])
        projected_points = world_points @ self.m_transformation
        projected_points = projected_points[:, :2] / projected_points[:, 2:3]
        map_matrix = projected_points.reshape(output_height, output_width, 2)

        return map_matrix

    def make_bev_image(self, image: np.ndarray) -> np.ndarray:
        bev_image = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        
        # np.modf를 이용해 소수부와 정수부(바닥 좌표)를 한 번에 계산
        frac, floor_coords = np.modf(self.valid_points)
        floor_coords = floor_coords.astype(np.int32)
        
        # 바닥 좌표와 보간을 위한 인덱스 계산
        x0 = floor_coords[:, 0]
        y0 = floor_coords[:, 1]
        x1 = x0 + 1
        y1 = y0 + 1
        
        # 이미지의 네 모서리 색상 추출
        left_top  = image[y0, x0]
        right_top = image[y0, x1]
        left_bottom = image[y1, x0]
        right_bottom = image[y1, x1]
        
        # 보간 계수
        dx = frac[:, 0:1]
        dy = frac[:, 1:2]
        inv_dx = 1 - dx
        inv_dy = 1 - dy
        
        # 각각의 기여도(가중치) 계산
        w_lt = inv_dx * inv_dy
        w_rt = dx * inv_dy
        w_lb = inv_dx * dy
        w_rb = dx * dy
        
        # 네 모서리 색상을 가중치에 따라 보간
        interpolated = (left_top  * w_lt +
                        right_top * w_rt +
                        left_bottom * w_lb +
                        right_bottom * w_rb).astype(np.uint8)
        
        # 유효한 위치(valid_mask)를 bev_image에 할당
        bev_image.reshape(-1, 3)[self.valid_mask] = interpolated
        
        return bev_image

    def image_to_bev_coord(self, image_point: tuple[float, float]) -> tuple[float, float]:
        u, v = image_point
        m_transformation = self.m_transformation
        A = np.array([
            [m_transformation[0, 2] * u - m_transformation[0, 0],
            m_transformation[1, 2] * u - m_transformation[1, 0]],
            [m_transformation[0, 2] * v - m_transformation[0, 1],
            m_transformation[1, 2] * v - m_transformation[1, 1]]
        ])
        B = np.array([
            (m_transformation[2, 2] * u - m_transformation[2, 0]) * self.camera_height,
            (m_transformation[2, 2] * v - m_transformation[2, 1]) * self.camera_height
        ])

        # 선형 방정식 A * [x_world, y_world]^T = B 풀기
        sol = np.linalg.solve(A, B)
        x_world, y_world = sol[0], sol[1]
        
        # Calculate BEV image coordinates using ranges and intervals
        col = (self.x_range[1] - x_world) / self.bev_pixel_interval[0]
        row = (-y_world - self.y_range[0]) / self.bev_pixel_interval[1]

        return (x_world, y_world), (row, col)

    def convert_world_to_bev(self, world_points):
        x = world_points[:, 0:1]
        x = (self.x_range[1] - x) / self.bev_pixel_interval[0]

        y = world_points[:, 1:2]
        y = ((self.y_range[1] - self.y_range[0]) / 2.0 - y) / self.bev_pixel_interval[1]

        return np.hstack([y, x])

    def convert_bev_to_world(self, bev_points):
        x = bev_points[:, 0:1]
        x = (self.y_range[1] - self.y_range[0]) / 2.0 - x * self.bev_pixel_interval[1]

        y = bev_points[:, 1:2]
        y = self.x_range[1] - y * self.bev_pixel_interval[0]

        return np.hstack([y, x])

    def convert_length_x_world_to_bev(self, world_length):
        return world_length / self.bev_pixel_interval[0]
    
    def convert_length_y_world_to_bev(self, world_length):
        return world_length / self.bev_pixel_interval[1]
    
    def convert_length_x_bev_to_world(self, bev_length):
        return bev_length * self.bev_pixel_interval[0]
    
    def convert_length_y_bev_to_world(self, bev_length):
        return bev_length * self.bev_pixel_interval[1]

if __name__ == "__main__":
    cap = cv2.VideoCapture("dataset/track.mp4")
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()

    ret, image = cap.read()
    if not ret:
        print("Error: Could not read frame")
        exit()

    image_shape = (image.shape[1], image.shape[0])
    print(image_shape)

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
    bev_pixel_interval = (0.02, 0.01)

    bev = BEV(image_shape, m_transformation, x_range, y_range, bev_pixel_interval, camera_height)

    map_matrix = bev.get_map_matrix()
    print(map_matrix.shape)

    projected_points = map_matrix.reshape(-1, 2)

    valid_mask = (0 <= projected_points[:, 0]) & (projected_points[:, 0] <= image_shape[0] - 1) & \
                 (0 <= projected_points[:, 1]) & (projected_points[:, 1] <= image_shape[1] - 1)
    valid_points = projected_points[valid_mask]
    
    bev_image = bev.make_bev_image(image)

    world_points = np.array([
        [2.0, 0.0],
        [5.0, 1.0],
        [4.0, -2.0]
    ])
    bev_points = bev.convert_world_to_bev(world_points)
    print(bev_points)
    world_points = bev.convert_bev_to_world(bev_points)
    print(world_points)

    for point in bev_points:
        print(point)
        cv2.circle(bev_image, point.astype(np.int32), 5, (255, 0, 0), -1)
    
    plt.imshow(bev_image)
    plt.show()
