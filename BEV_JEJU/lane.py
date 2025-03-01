import numpy as np
import cv2
import matplotlib.pyplot as plt

import calibration
from calibration import get_transformation_matrix
from bird_eye_view import BEV

class Window:
    def __init__(self, init_center, shape):
        self.center = init_center
        self.shape = shape
        self.half_shape = np.array([self.shape[0], self.shape[1]], dtype=np.float64) / 2.0
        self.direction = np.array([0.0, -1.0], dtype=np.float64)

    @property
    def xyxy(self):
        left_up = self.center - self.half_shape
        right_down = self.center + self.half_shape

        return left_up, right_down
    
    @property
    def xywh(self):
        return self.center, self.shape

    def arrange(self, prev_window):
        prev_direction = prev_window.direction
        self.center[0] = self.shape[1] * prev_direction[1] / prev_direction[0] # x : y = h : ?

    def crop(self, image):
        left_up, right_down = self.xyxy

        cropped = image[left_up[1]:right_down[1], left_up[0]:right_down[0]]

        return cropped
    
    @staticmethod
    def get_mean_x(cls, cropped):
        indices = np.where(cropped > 0)[1]
        mean = np.mean(indices)

        return mean
    
    @staticmethod
    def get_direction(cls, cropped):
        # 상하 반 나눠서 방향 벡터 계산
        half_h = cropped.shape[0] // 2

        top_points = np.argwhere(cropped[:half_h, :] > 0)
        bottom_points = np.argwhere(cropped[half_h:, :] > 0)
        bottom_points[:, 0] += half_h

        top_mean = np.mean(top_points, axis=0)
        bottom_mean = np.mean(bottom_points, axis=0)

        dir_vector = (top_mean - bottom_mean)[::-1] # 이미지는 (y, x), window는 (x, y)
        dir_vector = dir_vector / np.linalg.norm(dir_vector)

        return dir_vector




class SlidingWindows:
    def __init__(self, image_shape, lane_width, window_width, window_count):
        self.image_shape = image_shape
        self.lane_width = lane_width
        self.window_shape = np.array([window_width, image_shape[1] / window_count], dtype=np.float64)
        self.half_window_shape = self.window_shape / 2.0

        self.left_windows = []
        self.right_windows = []

        left_center = [(self.image_shape[0] - self.lane_width) / 2.0, self.image_shape[1] - self.half_window_shape[1]]
        right_center = [(self.image_shape[0] + self.lane_width) / 2.0, self.image_shape[1] - self.half_window_shape[1]]

        for i in range(window_count):
            self.left_windows.append(Window(np.array(left_center, dtype=np.float64), self.window_shape))
            self.right_windows.append(Window(np.array(right_center, dtype=np.float64), self.window_shape))

            left_center[1] -= self.window_shape[1]
            right_center[1] -= self.window_shape[1]

    def preprocess_image(self, image, valid_area=None):
        blured = cv2.GaussianBlur(image, (5, 5), 0)
        edged = cv2.Canny(blured, 50, 150)

        if valid_area is not None:
            edged[~valid_area] = 0

        return edged

    def align_windows(self, image):
        prev_left_mean = (self.image_shape[0] - self.lane_width) / 2.0
        prev_right_mean = (self.image_shape[0] + self.lane_width) / 2.0
        prev_win_distance = self.lane_width
        prev_win_left_dir = np.array([0.0, -1.0], dtype=np.float64)
        prev_win_right_dir = np.array([0.0, -1.0], dtype=np.float64)
        prev_win_center_dir = (prev_win_left_dir + prev_win_right_dir) / 2.0
        prev_win_center_x = self.image_shape[0] / 2.0

        for i, (left_window, right_window) in enumerate(zip(self.left_windows, self.right_windows)):
            prev_frame_distance = right_window.center[0] - left_window.center[0]
            prev_frame_left_dir = left_window.direction
            prev_frame_right_dir = right_window.direction

            left_cropped = left_window.crop(image)
            right_cropped = right_window.crop(image)

            # cropped 기준 x 좌표
            left_mean = Window.get_mean_x(left_cropped)
            right_mean = Window.get_mean_x(right_cropped)
            center_x = (left_mean + right_mean) / 2.0

            # left, right 사이 거리
            distance = right_mean - left_mean

            # cropped 기준 방향 벡터
            left_dir = Window.get_direction(left_cropped)
            right_dir = Window.get_direction(right_cropped)
            center_dir = (left_dir + right_dir) / 2.0

            # 이전 window와의 x 차이
            left_diff = left_mean - prev_left_mean
            right_diff = right_mean - prev_right_mean
            center_diff = center_x - prev_win_center_x








    def draw_windows(self, image):
        image_copy = image.copy()
        for left_window, right_window in zip(self.left_windows, self.right_windows):
            left_lu, left_rd = left_window.xyxy
            cv2.rectangle(image_copy, left_lu.astype(np.int32), left_rd.astype(np.int32), (255, 0, 0))

            right_lu, right_rd = right_window.xyxy
            cv2.rectangle(image_copy, right_lu.astype(np.int32), right_rd.astype(np.int32), (255, 0, 0))

        return image_copy


if __name__ == "__main__":
    cap = cv2.VideoCapture("dataset/video2.mp4")
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    
    ret, image = cap.read()
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

    w, h = image_shape

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    m_intrinsic = np.eye(4)
    m_intrinsic[:3, :3] = new_camera_matrix
    m_extrinsic = calibration.get_extrinsic_matrix(axis_rotation_radian, translation, rotation_radian)
    m_transformation = (m_intrinsic @ m_extrinsic).T

    x_range = (0.32, 1.7)
    y_range = (-3, 3)
    bev_pixel_interval = (0.005, 0.01)

    bev = BEV(image_shape, m_transformation, x_range, y_range, bev_pixel_interval, camera_height)

    lane_width = bev.convert_length_y_world_to_bev(0.5)
    window_width = bev.convert_length_y_world_to_bev(0.3)
    sliding_windows = SlidingWindows(bev.bev_shape, lane_width, window_width, 15)

    while True:
        ret, image = cap.read()
        if not ret:
            print("Error: Could not read frame") 
            break

        bev_image = bev.make_bev_image(image)

        preprocessed_image = sliding_windows.preprocess_image(bev_image, valid_area=bev.valid_area_margined)

        sw_image = sliding_windows.draw_windows(preprocessed_image)

        plt.imshow(sw_image)
        plt.show()
