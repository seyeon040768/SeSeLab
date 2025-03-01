import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml

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

    def arrange_by_prev_win(self, prev_win_x, prev_win_dir):
        next_dir = prev_win_dir + self.direction
        move_amount = self.shape[1] * next_dir[0] / np.abs(next_dir[1])
        move_amount = np.clip(move_amount, -self.shape[0], self.shape[0])
        new_x = prev_win_x + move_amount

        # TODO: 가중치 yaml 저장
        self.center[0] = new_x*0.7 + self.center[0]*0.3

    def crop(self, image):
        left_up, right_down = self.xyxy

        cropped = image[int(left_up[1]):int(right_down[1]), int(left_up[0]):int(right_down[0])]

        return cropped
    
    def update(self, x, direction):
        self.center[0] = x
        self.direction = direction
    
    def get_mean_x(self, image):
        cropped = self.crop(image)

        indices = np.where(cropped > 0)[1]
        if len(indices) == 0:
            return False, 0.0
        
        mean = np.mean(indices)

        return True, mean - self.half_shape[0]
    
    def get_direction(self, image):
        cropped = self.crop(image)

        mask = cropped > 0
        counts = np.sum(mask, axis=1)

        valid_rows = np.where(counts > 0)[0]
        if valid_rows.size < 2:
            return False, np.array([0.0, 0.0])
        
        mean_x = np.sum(mask[valid_rows] * np.arange(cropped.shape[1]), axis=1) / counts[valid_rows]
        mean_points = np.vstack([mean_x, valid_rows]).T

        dir_vectors = mean_points[:-1] - mean_points[1:]
        dir = np.mean(dir_vectors, axis=0)
        
        dir = dir / np.linalg.norm(dir)

        return True, dir





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

    def align_windows(self, image, distance_weights=(0.5, 0.5)):
        accumulated_dir = (self.left_windows[0].direction + self.right_windows[0].direction) / 2.0
        prev_left_x = (self.image_shape[0] - self.lane_width) / 2.0
        prev_right_x = (self.image_shape[0] + self.lane_width) / 2.0
        prev_left_dir = self.left_windows[0].direction
        prev_right_dir = self.right_windows[0].direction
        for i, (left_window, right_window) in enumerate(zip(self.left_windows, self.right_windows)):
            # 1. 초기 위치 설정
            left_window.arrange_by_prev_win(prev_left_x, accumulated_dir)
            right_window.arrange_by_prev_win(prev_right_x, accumulated_dir)

            arranged = self.draw_windows(image)

            # 2. 평균 x 계산 및 정렬(값 받으면 정렬, 못 받으면 그대로)
            left_mean_senti, left_mean = left_window.get_mean_x(image)
            right_mean_senti, right_mean = right_window.get_mean_x(image)
            left_window.center[0] += left_mean
            right_window.center[0] += right_mean

            mean_arranged = self.draw_windows(image)

            # 3. 방향 벡터 계산 TODO: 가중치화
            left_dir_weight, right_dir_weight = 0.5, 0.5
            left_dir_senti, left_dir = left_window.get_direction(image)
            if not left_dir_senti:
                left_dir = accumulated_dir
                left_dir_weight -= 0.2
                right_dir_weight += 0.2
            else:
                left_dir = left_dir * 0.7 + prev_left_dir * 0.3
            right_dir_senti, right_dir = right_window.get_direction(image)
            if not right_dir_senti:
                right_dir = accumulated_dir
                left_dir_weight += 0.2
                right_dir_weight -= 0.2
            else:
                right_dir = right_dir * 0.7 + prev_right_dir * 0.3
            center_dir = left_dir * left_dir_weight + right_dir * right_dir_weight

            accumulated_dir = (accumulated_dir + center_dir) / 2.0 # TODO: 가중치화
            accumulated_dir = accumulated_dir / np.linalg.norm(accumulated_dir)
            

            left_rectify = 0.0
            right_rectify = 0.0
            # 4. 거리 보정
            distance = right_window.center[0] - left_window.center[0]
            distance_diff = (distance - self.lane_width) / 2.0
            left_rectify += distance_diff * distance_weights[0] if left_mean_senti else distance_diff
            right_rectify += -distance_diff * distance_weights[1] if right_mean_senti else -distance_diff


            # 값 업데이트
            left_window.update(left_window.center[0] + left_rectify, left_dir)
            right_window.update(right_window.center[0] + right_rectify, right_dir)

            # prev 값 업데이트
            prev_left_x = left_window.center[0]
            prev_right_x = right_window.center[0]
            prev_left_dir = left_dir
            prev_right_dir = right_dir

            final = self.draw_windows(image)

            # plt.subplot(311)
            # plt.imshow(arranged)
            # plt.subplot(312)
            # plt.imshow(mean_arranged)
            # plt.subplot(313)
            # plt.imshow(final)
            # plt.show()






    def draw_windows(self, image, draw_arrow=True, draw_center=True):
        image_copy = image.copy()
        scale = self.window_shape[1]
        for left_window, right_window in zip(self.left_windows, self.right_windows):
            # 왼쪽 윈도우 사각형 그리기
            left_lu, left_rd = left_window.xyxy
            cv2.rectangle(image_copy, left_lu.astype(np.int32), left_rd.astype(np.int32), (255, 0, 0))

            # 오른쪽 윈도우 사각형 그리기
            right_lu, right_rd = right_window.xyxy
            cv2.rectangle(image_copy, right_lu.astype(np.int32), right_rd.astype(np.int32), (255, 0, 0))
            
            if draw_arrow:
                # 왼쪽 윈도우의 중앙과 방향벡터 (중심이 center에 위치)
                left_offset = left_window.direction * (scale / 2)
                left_start = (left_window.center - left_offset).astype(np.int32)
                left_end = (left_window.center + left_offset).astype(np.int32)
                cv2.arrowedLine(image_copy, tuple(left_start), tuple(left_end), (255, 0, 0), thickness=2, tipLength=0.2)
                
                
                # 오른쪽 윈도우의 중앙과 방향벡터 (중심이 center에 위치)
                right_offset = right_window.direction * (scale / 2)
                right_start = (right_window.center - right_offset).astype(np.int32)
                right_end = (right_window.center + right_offset).astype(np.int32)
                cv2.arrowedLine(image_copy, tuple(right_start), tuple(right_end), (255, 0, 0), thickness=2, tipLength=0.2)

            if draw_center:
                cv2.circle(image_copy, left_window.center.astype(np.int32), 2, (255, 0, 0), -1)
                cv2.circle(image_copy, right_window.center.astype(np.int32), 2, (255, 0, 0), -1)

        return image_copy


if __name__ == "__main__":
    with open("lane_parameter.yaml", encoding='UTF8') as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)
    print(parameters)

    cap = cv2.VideoCapture("dataset/video2.mp4")
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    
    ret, image = cap.read()
    if not ret:
        print("Error: Could not read frame")
        exit()

    image_shape = (image.shape[1], image.shape[0])
    print("image shape:", image_shape)

    camera_matrix = np.array(parameters["camera_matrix"], dtype=np.float64).reshape((3, 3))
    dist_coeffs = np.array(parameters["dist_coeffs"], dtype=np.float64)

    axis_rotation_degree = tuple(parameters["axis_rotation_degree"])
    translation = tuple(parameters["translation"])
    rotation_degree = tuple(parameters["rotation_degree"])
    fov_degree = parameters["fov_degree"]
    camera_height = parameters["camera_height"]

    axis_rotation_radian = np.deg2rad(axis_rotation_degree)
    rotation_radian = np.deg2rad(rotation_degree)
    fov_radian = np.deg2rad(fov_degree)

    w, h = image_shape
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    m_intrinsic = np.eye(4)
    m_intrinsic[:3, :3] = new_camera_matrix
    m_extrinsic = calibration.get_extrinsic_matrix(axis_rotation_radian, translation, rotation_radian)
    m_transformation = (m_intrinsic @ m_extrinsic).T

    x_range = tuple(parameters["x_range"])
    y_range = tuple(parameters["y_range"])
    bev_pixel_interval = tuple(parameters["bev_pixel_interval"])

    bev = BEV(image_shape, m_transformation, x_range, y_range, bev_pixel_interval, camera_height)
    print("bev shape:", bev.bev_shape)

    lane_width = bev.convert_length_y_world_to_bev(parameters["lane_width"])
    window_width = bev.convert_length_y_world_to_bev(parameters["window_width"])
    window_count = parameters["window_count"]
    sliding_windows = SlidingWindows(bev.bev_shape, lane_width, window_width, window_count)

    while True:
        ret, image = cap.read()
        if not ret:
            print("Error: Could not read frame") 
            break

        bev_image = bev.make_bev_image(image)

        preprocessed_image = sliding_windows.preprocess_image(bev_image, valid_area=bev.valid_area_margined)
        sliding_windows.align_windows(preprocessed_image)

        sw_image = sliding_windows.draw_windows(preprocessed_image)

        bev_image = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)

        vis_image_1 = np.hstack([bev_image, preprocessed_image])
        vis_image_2 = np.hstack([sw_image, np.zeros_like(sw_image)])
        vis_image = np.vstack([vis_image_1, vis_image_2])

        cv2.imshow("img", vis_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
