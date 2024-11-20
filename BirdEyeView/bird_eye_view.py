import numpy as np
import cv2
import matplotlib.pyplot as plt
import calibration
from copy import deepcopy

image_path = "./um_000012.png"
image = cv2.imread(image_path)
image_shape = (image.shape[1], image.shape[0])

rotation_degree = (90, -90, 0)
# translation = (0.06, -0.08, -0.27)
translation = (0.06, -7.631618000000e-02, -2.717806000000e-01)
fov_degree = 85.7
camera_height = 1.65
m_transformation = calibration.get_transformation_matrix(image_shape, np.deg2rad(fov_degree), np.deg2rad(rotation_degree), translation).T

x_range = (6.3, 50)
y_range = (-10, 10)

bev_pixel_interval = (0.05, 0.05)

x_samples = np.arange(x_range[0], x_range[1], bev_pixel_interval[0])
y_samples = np.arange(y_range[0], y_range[1], bev_pixel_interval[1])
output_height = len(x_samples)
output_width = len(y_samples)

map_matrix = np.zeros((output_height, output_width, 2), dtype=np.float32)
for i, x in enumerate(x_samples):
    for j, y in enumerate(y_samples):
        world_point = np.array([x, y, -camera_height, 1])
        projected_point = world_point @ m_transformation
        projected_point = projected_point[:2] / projected_point[2]

        map_matrix[output_height - i - 1, output_width - j - 1] = projected_point

area_image = deepcopy(image)
for i in range(map_matrix.shape[0]):
    for j in range(map_matrix.shape[1]):
        projected_point = map_matrix[i, j]
        if 0 <= projected_point[0] < image_shape[0] and 0 <= projected_point[1] < image_shape[1]:
            cv2.circle(area_image, (int(projected_point[0]), int(projected_point[1])), 1, (0, 0, 255), -1)

bev_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
for i in range(map_matrix.shape[0]):
    for j in range(map_matrix.shape[1]):
        projected_point = map_matrix[i, j]
        if 0 <= projected_point[0] <= image_shape[0] - 1 and 0 <= projected_point[1] <= image_shape[1] - 1:
            floored_projected_point = np.floor(projected_point).astype(np.int32)
            dist = projected_point - floored_projected_point
            
            # bilinear interpolation
            left_top_color = image[floored_projected_point[1], floored_projected_point[0]]
            right_top_color = image[floored_projected_point[1], floored_projected_point[0] + 1]
            left_bottom_color = image[floored_projected_point[1] + 1, floored_projected_point[0]]
            right_bottom_color = image[floored_projected_point[1] + 1, floored_projected_point[0] + 1]

            top_interpolated_color = left_top_color * (1 - dist[0]) + right_top_color * dist[0]
            bottom_interpolated_color = left_bottom_color * (1 - dist[0]) + right_bottom_color * dist[0]
            interpolated_color = top_interpolated_color * (1 - dist[1]) + bottom_interpolated_color * dist[1]

            bev_image[i, j] = interpolated_color

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(cv2.cvtColor(area_image, cv2.COLOR_BGR2RGB))
plt.title('Area Image')
plt.axis('off')

plt.subplot(133)
plt.imshow(cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB))
plt.title('Bird\'s Eye View')
plt.axis('off')

plt.tight_layout()
plt.savefig('bev_result.png')

plt.show()

