import numpy as np
import cv2
import matplotlib.pyplot as plt

class SlidingWindow:
    def __init__(self, shape, position):
        self.shape = shape
        self.position = position
        self.lane = np.array([0, 0])

    def get_vertices(self):
        x_start = int(self.position[0] - self.shape[0] / 2)
        x_end = int(self.position[0] + self.shape[0] / 2)
        y_start = int(self.position[1] - self.shape[1] / 2)
        y_end = int(self.position[1] + self.shape[1] / 2)
        
        return ((x_start, y_start), (x_end, y_end))

class SlidingWindowHolder:
    def __init__(self, image, window_count, window_width, offset=50):
        self.image = image
        self.image_shape = np.array((image.shape[1], image.shape[0]))
        self.window_count = window_count
        self.window_shape = np.array((window_width, self.image_shape[1] / window_count))

        self.left_windows = []
        self.right_windows = []

        left_position = self.image_shape[0] / 2 - offset
        right_position = self.image_shape[0] - left_position
        height_position = self.image_shape[1] - self.window_shape[1] / 2

        for _ in range(window_count):
            self.left_windows.append(SlidingWindow(self.window_shape, np.array((left_position, height_position))))
            self.right_windows.append(SlidingWindow(self.window_shape, np.array((right_position, height_position))))
            height_position -= self.window_shape[1]

    def get_lane(self):
        for i, left_window in enumerate(self.left_windows):
            start_pos, end_pos = left_window.get_vertices()

            area = self.image[start_pos[1]:end_pos[1], start_pos[0]:end_pos[0]]

            lane_indices = np.where(area > 0)
            lane_center = None
            if lane_indices[0].size > 0:
                lane_center = np.mean(lane_indices, axis=1)[::-1]
            else:
                lane_center = left_window.shape / 2

            lane_center -= left_window.shape / 2
            self.left_windows[i].position[0] += lane_center[0]

        for i, right_window in enumerate(self.right_windows):
            start_pos, end_pos = right_window.get_vertices()

            area = self.image[start_pos[1]:end_pos[1], start_pos[0]:end_pos[0]]

            lane_indices = np.where(area > 0)
            lane_center = None
            if lane_indices[0].size > 0:
                lane_center = np.mean(lane_indices, axis=1)[::-1]
            else:
                lane_center = right_window.shape / 2

            lane_center -= right_window.shape / 2
            self.right_windows[i].position[0] += lane_center[0]

    def print_windows(self, image=None):
        image = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
            
        
        for window in self.left_windows + self.right_windows:
            start_point, end_point = window.get_vertices()
            
            cv2.rectangle(image, start_point, end_point, (255, 255, 255), 1)
            
            # lane_point = window.lane + window.position
            # cv2.circle(image, (int(lane_point[0]), int(lane_point[1])), 5, (0, 0, 255), -1)

        for left_window, right_window in zip(self.left_windows, self.right_windows):
            lane_center = (left_window.position + right_window.position) / 2
            cv2.circle(image, (int(lane_center[0]), int(lane_center[1])), 5, (0, 0, 255), -1)
            
        return image

if __name__ == "__main__":
    from lane_detection import process_image
    from calibration import get_transformation_matrix

    axis_rotation = (90, -90, 0)
    translation = (0, 0, 0) # camera 위치 기준
    rotation = (5.16, 0, 0)
    fov_degree = 60
    camera_height = 1.5
    image_shape = (1280, 720)

    x_range = (10, 50)
    y_range = (-20, 20)
    bev_pixel_interval = (0.05, 0.05)

    result_shape = np.ceil(((y_range[1] - y_range[0]) / bev_pixel_interval[1], (x_range[1] - x_range[0]) / bev_pixel_interval[0])).astype(np.int32)

    x_roi = (300, 500)
    m_transformation = get_transformation_matrix(image_shape, np.deg2rad(fov_degree), np.deg2rad(axis_rotation), translation, np.deg2rad(rotation)).T

    bev_image, edge_roi = process_image("1.jpg", m_transformation, x_range, y_range, bev_pixel_interval, camera_height, x_roi)

    holder = SlidingWindowHolder(edge_roi, 10, 40, offset=35)

    holder.get_lane()
    result_image = holder.print_windows()
    mask = result_image > 0
    bev_image[:, 300:500, :][mask] = result_image[mask]
    bev_image = np.clip(bev_image, 0, 255).astype(np.uint8)
    print(result_image.shape)
    plt.figure(figsize=(10, 5))
    plt.imshow(bev_image)
    plt.title("Sliding Windows")
    plt.show()
