import numpy as np

def sad(block1: np.ndarray, block2: np.ndarray) -> float:
    diff = block1 - block2
    diff_abs = np.abs(diff)
    return np.sum(diff_abs)

def ssd(block1: np.ndarray, block2: np.ndarray) -> float:
    diff = block1 - block2
    diff_sq = np.square(diff)
    return np.sum(diff_sq)

def ncc(block1: np.ndarray, block2: np.ndarray) -> float:
    block1_mean = np.mean(block1)
    block2_mean = np.mean(block2)

    block1_diff = block1 - block1_mean
    block2_diff = block2 - block2_mean

    block1_diff_sq = np.square(block1_diff)
    block2_diff_sq = np.square(block2_diff)

    numerator = np.sum(block1_diff * block2_diff)
    denominator = np.sqrt(np.sum(block1_diff_sq) * np.sum(block2_diff_sq))

    if denominator == 0:
        return 0
    return numerator / denominator

def census_transform(block1: np.ndarray, block2: np.ndarray) -> float:
    center_pos_y = block1.shape[0] // 2
    center_pos_x = block1.shape[1] // 2

    block1_diff = block1 - block1[center_pos_y, center_pos_x]
    block2_diff = block2 - block2[center_pos_y, center_pos_x]

    block1_diff = np.where(block1_diff < 0, 1, 0)
    block2_diff = np.where(block2_diff < 0, 1, 0)

    diff = block1_diff - block2_diff
    
    return np.count_nonzero(diff)