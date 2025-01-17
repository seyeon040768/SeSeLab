import cv2
import numpy as np
from numba import njit, prange
from tqdm import tqdm

def compute_initial_cost(left, right, max_disparity, reverse=False):
    height, width = left.shape
    cost_volume = np.zeros((height, width, max_disparity), dtype=np.float32)

    if not reverse:
        for d in range(max_disparity):
            shifted_right = np.roll(right, d, axis=1)
            shifted_right[:, :d] = 255
            cost = np.abs(left - shifted_right)
            cost_volume[:, :, d] = cost
    else:
        for d in range(max_disparity):
            shifted_right = np.roll(right, -d, axis=1)
            shifted_right[:, -d:] = 255
            cost = np.abs(left - shifted_right)
            cost_volume[:, :, d] = cost

    return cost_volume

def aggregate_cost(cost_volume, P1, P2):
    height, width, max_disparity = cost_volume.shape

    direction_map = np.array([
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 1),
        (1, 1),
        (1, -1),
        (-1, -1)
    ])

    aggregated_cost = np.zeros_like(cost_volume, dtype=np.float32)

    for dy, dx in tqdm(direction_map):
        direction_cost = np.zeros_like(cost_volume, dtype=np.float32)

        if dy >= 0 and dx >= 0:
            y_range = range(height)
            x_range = range(width)
        elif dy >= 0 and dx < 0:
            y_range = range(height)
            x_range = range(width-1, -1, -1)
        elif dy < 0 and dx >= 0:
            y_range = range(height-1, -1, -1)
            x_range = range(width)
        else:
            y_range = range(height-1, -1, -1)
            x_range = range(width-1, -1, -1)

        for y in y_range:
            for x in x_range:
                prev_y = y - dy
                prev_x = x - dx

                if prev_y < 0 or prev_y >= height or prev_x < 0 or prev_x >= width:
                    direction_cost[y, x, :] = cost_volume[y, x, :]
                else:
                    cost = cost_volume[y, x, :]

                    prev_cost = direction_cost[prev_y, prev_x, :]
                    prev_cost_d_minus = np.empty_like(prev_cost)
                    prev_cost_d_minus[1:] = prev_cost[:-1]
                    prev_cost_d_minus[0] = np.inf

                    prev_cost_d_plus = np.empty_like(prev_cost)
                    prev_cost_d_plus[:-1] = prev_cost[1:]
                    prev_cost_d_plus[-1] = np.inf

                    prev_min_cost = np.min(prev_cost)

                    min_cost = np.minimum(prev_cost, prev_cost_d_minus + P1)
                    min_cost = np.minimum(min_cost, prev_cost_d_plus + P1)
                    min_cost = np.minimum(min_cost, prev_min_cost + P2)

                    direction_cost[y, x, :] = cost + min_cost - prev_min_cost
        
        aggregated_cost += direction_cost

    return aggregated_cost

def select_disparity(aggregated_cost):
    disparity_map = np.argmin(aggregated_cost, axis=2).astype(np.int32)
    return disparity_map

def compute_sgm(left, right, max_disparity, P1=10, P2=120, reverse=True):
    cost_volume = compute_initial_cost(left, right, max_disparity, reverse)

    aggregated_cost = aggregate_cost(cost_volume, P1, P2)

    disparity_map = select_disparity(aggregated_cost)

    return disparity_map


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    left = cv2.imread('data/data_left.png', cv2.IMREAD_GRAYSCALE).astype(np.int32)
    right = cv2.imread('data/data_right.png', cv2.IMREAD_GRAYSCALE).astype(np.int32)

    left = cv2.imread('data/left.png', cv2.IMREAD_GRAYSCALE)
    right = cv2.imread('data/right.png', cv2.IMREAD_GRAYSCALE)
    left = cv2.resize(left, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC).astype(np.int32)
    right = cv2.resize(right, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC).astype(np.int32)

    np.set_printoptions(threshold=1000000)

    disparity_map = compute_sgm(left, right, 32)

    plt.figure(figsize=(20, 5))

    plt.subplot(121)
    plt.imshow(disparity_map, cmap='jet', vmin=0, vmax=31)
    plt.colorbar(label='Cost')
    plt.title('Disparity Map')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.addWeighted(left, 0.5, right, 0.5, 0), cmap='gray')
    plt.title('Merged Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()