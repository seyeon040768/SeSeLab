import cv2
import numpy as np
import matplotlib.pyplot as plt

from matching_algorithms.block_matching import compute_block_matching
from matching_algorithms.sgm import compute_sgm
from triangulation import triangulation3d, compute_distance_map

def compare_left_right_disparity(disparity_map_left, disparity_map_right, tolerance=3):
    for y in range(disparity_map_left.shape[0]):
        for x in range(disparity_map_left.shape[1]):
            d = disparity_map_left[y, x]
            right_x = x - d

            if right_x < 0:
                continue

            d_right = disparity_map_right[y, right_x]

            if np.abs(d - d_right) > tolerance:
                continue

            disparity_map_left[y, x] = (d + d_right) / 2

    return disparity_map_left

# simple shape image
# left = cv2.imread('data/data_left.png', cv2.IMREAD_GRAYSCALE).astype(np.int32)
# right = cv2.imread('data/data_right.png', cv2.IMREAD_GRAYSCALE).astype(np.int32)

# venus image
left = cv2.imread('data/left.png', cv2.IMREAD_GRAYSCALE)
right = cv2.imread('data/right.png', cv2.IMREAD_GRAYSCALE)

# cones image
# left = cv2.imread('data/cones_left.png', cv2.IMREAD_GRAYSCALE)
# right = cv2.imread('data/cones_right.png', cv2.IMREAD_GRAYSCALE)

# stair image
# left = cv2.imread('data/left_stair.jpg', cv2.IMREAD_GRAYSCALE)
# right = cv2.imread('data/right_stair.jpg', cv2.IMREAD_GRAYSCALE)

resize_ratio = 1
if resize_ratio != 1:
    left = cv2.resize(left, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_CUBIC).astype(np.int32)
    right = cv2.resize(right, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_CUBIC).astype(np.int32)

# disparity_map = compute_block_matching(left, right, 3, 64, method="ncc")
# disparity_map_left = compute_sgm(left, right, 64*10, 10, 120, False)
# disparity_map_right = compute_sgm(right, left, 64*10, 10, 120, True)

# np.save('disparity_map_venus_left_1_p.npy', disparity_map_left)
# np.save('disparity_map_venus_right_1_p.npy', disparity_map_right)

disparity_map_left = np.load('disparity_map_venus_left_1.npy')
disparity_map_right = np.load('disparity_map_venus_right_1.npy')

# disparity_map_left = compare_left_right_disparity(disparity_map_left, disparity_map_right)

# baseline은 160mm = 0.16m
baseline = 0.16  # meters
focal_length = 3740 * resize_ratio  # pixels

# 거리 맵 계산
distance_map_left = compute_distance_map(disparity_map_left, focal_length, baseline)
distance_map_right = compute_distance_map(disparity_map_right, focal_length, baseline)

# np.set_printoptions(threshold=10000000)
# print(disparity_map_left)
# print(disparity_map_right)

min_distance = 0
max_distance = 6

clipped_distance_left = max_distance - np.clip(distance_map_left, min_distance, max_distance) + min_distance
clipped_distance_right = max_distance - np.clip(distance_map_right, min_distance, max_distance) + min_distance

plt.figure(figsize=(15, 10))
plt.subplot(231)
plt.imshow(left, cmap='gray')
plt.title('Left Image')
plt.axis('off')

plt.subplot(232)
plt.imshow(disparity_map_left, cmap='jet', vmin=0, vmax=64*10-1)
plt.title('Disparity Map')
plt.axis('off')

plt.subplot(233)
plt.imshow(clipped_distance_left, cmap='gray')
plt.title('Distance Map')
plt.axis('off')

plt.subplot(234)
plt.imshow(right, cmap='gray')
plt.title('Right Image')
plt.axis('off')

plt.subplot(235)
plt.imshow(disparity_map_right, cmap='jet', vmin=0, vmax=64*10-1)
plt.title('Disparity Map')
plt.axis('off')

plt.subplot(236)
plt.imshow(clipped_distance_right, cmap='gray')
plt.title('Distance Map')
plt.axis('off')

plt.show()








