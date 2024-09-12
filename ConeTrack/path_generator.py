import numpy as np
import matplotlib.pyplot as plt
import copy

from typing import Tuple, Callable

def visualize_path(ax: plt.Axes, 
                   left_cones_sorted: np.ndarray, right_cones_sorted: np.ndarray, 
                   m_left_direction: np.ndarray, m_right_direction: np.ndarray, 
                   m_left_perpendicular: np.ndarray, m_right_perpendicular: np.ndarray,
                   left_paths: np.ndarray, right_paths: np.ndarray, paths: np.ndarray):
    """Visualize path

    Args:
        ax (plt.Axes): matplotlib axes
        left_cones_sorted (np.ndarray)
        right_cones_sorted (np.ndarray)
        m_left_direction (np.ndarray)
        m_right_direction (np.ndarray)
        m_left_perpendicular (np.ndarray)
        m_right_perpendicular (np.ndarray)
        left_paths (np.ndarray)
        right_paths (np.ndarray)
        paths (np.ndarray)
    """
    ax.scatter(left_cones_sorted[:, 0], left_cones_sorted[:, 1], c="b")
    ax.scatter(right_cones_sorted[:, 0], right_cones_sorted[:, 1], c="r")
    for i, _ in enumerate(m_left_direction):
        ax.arrow(left_cones_sorted[i, 0], left_cones_sorted[i, 1], m_left_direction[i, 0], m_left_direction[i, 1], head_width=0.15, head_length=0.15, length_includes_head=True, overhang=1)
    for i, _ in enumerate(m_right_direction):
        ax.arrow(right_cones_sorted[i, 0], right_cones_sorted[i, 1], m_right_direction[i, 0], m_right_direction[i, 1], head_width=0.15, head_length=0.15, length_includes_head=True, overhang=1)
    for i, _ in enumerate(m_left_perpendicular):
        ax.arrow((left_cones_sorted[i, 0]+left_cones_sorted[i+1, 0])/2, (left_cones_sorted[i, 1]+left_cones_sorted[i+1, 1])/2, m_left_perpendicular[i, 0], m_left_perpendicular[i, 1], head_width=0.15, head_length=0.15, length_includes_head=True, overhang=1)
    for i, _ in enumerate(m_right_perpendicular):
        ax.arrow((right_cones_sorted[i, 0]+right_cones_sorted[i+1, 0])/2, (right_cones_sorted[i, 1]+right_cones_sorted[i+1, 1])/2, m_right_perpendicular[i, 0], m_right_perpendicular[i, 1], head_width=0.15, head_length=0.15, length_includes_head=True, overhang=1)
    ax.scatter(left_paths[:, 0], left_paths[:, 1], c="y")
    ax.scatter(right_paths[:, 0], right_paths[:, 1], c="y")
    ax.scatter(paths[:, 0], paths[:, 1], c="black")


def detect_outlier_IQR(data: np.ndarray, iqr_weight: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """Detect outlier with IQR

    Args:
        data (np.ndarray): 1-dim array
        iqr_weight (float): iqr multiple weight, Defaults to 1.5

    Returns:
        Tuple[np.ndarray, np.ndarray]: ([lower_outlier_indexes], [upper_outlier_indexes])
    """    
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1

    lower_threshold = q1 - iqr_weight * iqr
    upper_threshold = q3 + iqr_weight * iqr

    return np.where(data < lower_threshold)[0], np.where(data > upper_threshold)[0]


def detect_outlier_m_z_score(data: np.ndarray, threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
    """Detect outlier with Modified Z-Score

    Args:
        data (np.ndarray): 1-dim array
        threshold (float): threshold for Modified Z-Score, Defaults to 3.5

    Returns:
        Tuple[np.ndarray, np.ndarray]: ([lower_outlier_indexes], [upper_outlier_indexes])
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad
    return np.where(modified_z_scores < -threshold)[0], np.where(modified_z_scores > threshold)[0]


def insert_missing_cones(left_cones: np.ndarray, right_cones: np.ndarray, 
                         left_dist: np.ndarray, right_dist: np.ndarray, outlier_function: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """Detect and insert missing cones

    Args:
        left_cones (np.ndarray): (x, y) centroid points of left side cones, shape is (n, 2)
        right_cones (np.ndarray): (x, y) centroid points of right side cones, shape is (n, 2)
        left_dist (np.ndarray): distance of sequential left cones
        right_dist (np.ndarray): distance of sequential right cones
        outlier_function (Callable): function that detects outliers

    Returns:
        Tuple[np.ndarray, np.ndarray]: (left_cones, right_cones)
    """
    left_cones_copy = copy.deepcopy(left_cones)
    right_cones_copy = copy.deepcopy(right_cones)

    _, left_upper_outlier_index = outlier_function(left_dist)
    _, right_upper_outlier_index = outlier_function(right_dist)

    left_dist_mean = np.mean(np.delete(left_dist, left_upper_outlier_index))
    right_dist_mean = np.mean(np.delete(right_dist, right_upper_outlier_index))

    for i in reversed(left_upper_outlier_index):
        additional_cones = np.linspace(left_cones_copy[i], left_cones_copy[i+1], int(np.round(left_dist[i] / left_dist_mean)) - 1 + 2)[1:-1]
        left_cones_copy = np.insert(left_cones_copy, i+1, additional_cones, axis=0)
    for i in reversed(right_upper_outlier_index):
        additional_cones = np.linspace(right_cones_copy[i], right_cones_copy[i+1], int(np.round(right_dist[i] / right_dist_mean)) - 1 + 2)[1:-1]
        right_cones_copy = np.insert(right_cones_copy, i+1, additional_cones, axis=0)

    return left_cones_copy, right_cones_copy
    

def sort_cones(cones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort the cones along the track.

    Args:
        cones (np.ndarray): (x, y) centroid points of cones, shape is (n, 2)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (sorted cones, distance of sequential cones)
    """
    cones_copy = copy.deepcopy(cones)

    dist = []

    pivot = np.array([0, 0])
    for i in range(cones_copy.shape[0] - 1):
        dist_from_pivot = np.linalg.norm(cones_copy[i:] - pivot, axis=1)
        min_index = np.argmin(dist_from_pivot)
        dist.append(dist_from_pivot[min_index])
        cones_copy[[i, i+min_index]] = cones_copy[[i+min_index, i]]
        pivot = cones_copy[i]

    dist.append(np.linalg.norm(cones_copy[-1] - pivot))

    return cones_copy, np.array(dist[1:])
        

def generate_path(left_cones: np.ndarray, right_cones: np.ndarray, track_width: float = 0, b_visualize: bool = True) -> Tuple[bool, np.ndarray]:
    """Generate path

    Args:
        left_cones (np.ndarray): (x, y) centroid points of left side cones, shape is (n, 2)
        right_cones (np.ndarray): (x, y) centroid points of right side cones, shape is (n, 2)
        track_width (float, optional): track width, defaults to auto calculate

    Returns:
        Tuple[bool, np.ndarray]: (is success, path marked with several points)
    """
    # TODO(Research): Use Moore-Penrose Inverse Matrix for Linear Regression

    left_cones_roi = left_cones[np.where(left_cones[:, 0] > 0)]
    right_cones_roi = right_cones[np.where(right_cones[:, 0] > 0)]

    if left_cones_roi.size == 0 or right_cones_roi.size == 0:
        return False, np.array([])

    left_cones_sorted, left_dist = sort_cones(left_cones_roi)
    right_cones_sorted, right_dist = sort_cones(right_cones_roi)

    track_width = track_width if track_width != 0 else np.linalg.norm(left_cones_sorted[0] - right_cones_sorted[0])

    left_cones_sorted, right_cones_sorted = insert_missing_cones(left_cones_sorted, right_cones_sorted, left_dist, right_dist, detect_outlier_m_z_score)

    m_left_direction = left_cones_sorted[1:] - left_cones_sorted[:-1]
    m_right_direction = right_cones_sorted[1:] - right_cones_sorted[:-1]

    m_left_perpendicular = m_left_direction[:, [1, 0]] / np.linalg.norm(m_left_direction, axis=1)[:, np.newaxis]
    m_left_perpendicular[:, 1] = -m_left_perpendicular[:, 1]
    m_right_perpendicular = m_right_direction[:, [1, 0]] / np.linalg.norm(m_right_direction, axis=1)[:, np.newaxis]
    m_right_perpendicular[:, 0] = -m_right_perpendicular[:, 0]

    left_paths = (left_cones_sorted[1:] + left_cones_sorted[:-1]) / 2 + m_left_perpendicular * (track_width / 2)
    right_paths = (right_cones_sorted[1:] + right_cones_sorted[:-1]) / 2 + m_right_perpendicular * (track_width / 2)
    paths = None
    if left_paths.shape[0] > right_paths.shape[0]:
        paths = (left_paths[:right_paths.shape[0]] + right_paths) / 2
        paths = np.vstack([paths, left_paths[right_paths.shape[0]:]])
    elif left_paths.shape[0] < right_paths.shape[0]:
        paths = (left_paths + right_paths[:left_paths.shape[0]]) / 2
        paths = np.vstack([paths, right_paths[left_paths.shape[0]:]])
    else:
        paths = (left_paths + right_paths) / 2

    if b_visualize:
        ax = plt.subplot()
        visualize_path(ax, left_cones_sorted, right_cones_sorted, m_left_direction, m_right_direction, m_left_perpendicular, m_right_perpendicular, left_paths, right_paths, paths)
        plt.axis("equal")
        plt.show()

    return True, paths


# left_cones = np.array([[0.75, 1.25], [1, 2], [2.5, 3.5], [1.5, 2.5], [2, 3], [0.5, 0.5], [5, 6], [3, 4]])
# left_cones = np.array([[-1, 0], [0.01, 0], [1, 0], [5, 0], [6, -1]])
# right_cones = left_cones + np.array([0.3, -1])
# right_cones = left_cones + np.array([1, -0.5])
# right_cones = right_cones[:-1]

# left_cones = np.array([(174, 231), (205, 215), (311, 192), (330, 222), (338, 260), (358, 289), (374, 310), (406, 337), (460, 332), (499, 309), (519, 281), (538, 241), (564, 195), (577, 162)])
# right_cones = np.array([(157, 197), (194, 174), (232, 151), (268, 132), (307, 126), (343, 153), (363, 193), (375, 231), (386, 262), (415, 289), (465, 277), (494, 251), (517, 211), (536, 165), (549, 121)])

# generate_path(left_cones, right_cones)