import numpy as np
from utils import sad, ssd, ncc

def compute_block_matching(left: np.ndarray, right: np.ndarray, block_size: int, disparity_range: int, method: str = 'sad') -> np.ndarray:
    """Compute the block matching disparity map between two images.

    Args:
        left (np.ndarray): The left image.
        right (np.ndarray): The right image.
        block_size (int): The size of the block.
        disparity_range (int): The range of the disparity.
        method (str): Matching method ('sad', 'ssd', or 'ncc'). Defaults to 'sad'.

    Returns:
        np.ndarray: The disparity map.
    """
    assert left.shape == right.shape, "Images must have the same shape"
    assert left.ndim == 2, "Images must be grayscale"
    assert block_size > 0, "Block size must be greater than 0"
    assert disparity_range > 0, "Disparity range must be greater than 0"

    height, width = left.shape

    disparity_map = np.zeros((height, width), dtype=np.float32)

    half_block_size = block_size // 2

    img1_padded = np.pad(left, ((half_block_size, half_block_size), (half_block_size, half_block_size)), mode='constant')
    img2_padded = np.pad(right, ((half_block_size, half_block_size), (half_block_size, half_block_size)), mode='constant')

    for y in range(half_block_size, height - half_block_size):
        for x in range(half_block_size, width - half_block_size):
            best_cost = float('inf')
            best_disparity = 0

            block1 = img1_padded[y - half_block_size:y + half_block_size + 1, x - half_block_size:x + half_block_size + 1]

            for d in range(disparity_range):
                img2_x = x - d
                img2_x_bound = (img2_x - half_block_size, img2_x + half_block_size)

                if img2_x_bound[0] < 0 or img2_x_bound[1] >= width:
                    continue

                block2 = img2_padded[y - half_block_size:y + half_block_size + 1, img2_x_bound[0]:img2_x_bound[1] + 1]

                # Select matching method
                if method == 'sad':
                    current_cost = sad(block1, block2)
                elif method == 'ssd':
                    current_cost = ssd(block1, block2)
                elif method == 'ncc':
                    current_cost = -ncc(block1, block2)
                else:
                    raise ValueError(f"Unknown matching method: {method}")

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_disparity = d

            disparity_map[y, x] = best_disparity

    return disparity_map


