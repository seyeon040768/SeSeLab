from typing import List, Tuple, Union, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import enum

class ELabel(enum.Enum):
    UNKNOWN = -1
    RED = 0
    YELLOW = 1
    GREEN = 2
    LEFT = 3
    RED_LEFT = 4
    YELLOW_LEFT = 5

    @classmethod
    def from_str(cls, label_str: str) -> 'ELabel':
        return cls[label_str.upper()]
    
    @classmethod
    def from_mask(cls, mask: Tuple[bool, bool, bool, bool]) -> 'ELabel':
        num = (mask[0] << 3) | (mask[1] << 2) | (mask[2] << 1) | (mask[3] << 0)
        
        mask_map = (cls.UNKNOWN, cls.GREEN, cls.UNKNOWN, cls.LEFT,
                   cls.YELLOW, cls.UNKNOWN, cls.YELLOW_LEFT, cls.UNKNOWN,
                   cls.RED, cls.UNKNOWN, cls.RED_LEFT, cls.UNKNOWN,
                   cls.UNKNOWN, cls.UNKNOWN, cls.UNKNOWN, cls.UNKNOWN)
        
        return mask_map[num]
    
    @classmethod
    def to_str(cls, label: 'ELabel') -> str:
        return label.name.lower()

    @classmethod
    def to_mask(cls, label: 'ELabel') -> Tuple[bool, bool, bool, bool]:
        if label == cls.UNKNOWN:
            return (False, False, False, False)
        
        mask_map = (cls.UNKNOWN, cls.GREEN, cls.UNKNOWN, cls.LEFT,
                   cls.YELLOW, cls.UNKNOWN, cls.UNKNOWN, cls.YELLOW_LEFT,
                   cls.RED, cls.UNKNOWN, cls.UNKNOWN, cls.RED_LEFT,
                   cls.UNKNOWN, cls.UNKNOWN, cls.UNKNOWN, cls.UNKNOWN)
        
        num = mask_map.index(label)
        
        return (
            bool((num >> 3) & 1),
            bool((num >> 2) & 1),
            bool((num >> 1) & 1),
            bool((num >> 0) & 1) 
        )

def try_read_image_label(path: str) -> Tuple[bool, Union[np.ndarray, List], Union[ELabel, int]]:
    if not os.path.exists(path):
        return False, [], 0
    
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    filename = path.replace('\\', '/').split('/')[-1]
    label_str = filename.split('_')[:-1]
    label_str = '_'.join(label_str)
    label = ELabel.from_str(label_str)
    return True, image, label

def read_image_label(data_dir: str) -> Tuple[Tuple[np.ndarray, ELabel], ...]:
    image_label_lst = []
    for label in ELabel:
        label_str = ELabel.to_str(label)

        i = 1
        while True:
            path = os.path.join(data_dir, f"{label_str}_{i}.png")

            ret, image, label = try_read_image_label(path)
            if not ret:
                break

            image_label_lst.append((image, label))
            i += 1

    return tuple(image_label_lst)

def get_active_traffic_lights(light_means: np.ndarray, weight: float = 0.5, threshold = 0.1) -> np.ndarray:
    median = np.median(light_means) # 4개의 평균 밝기 중 중앙값
    threshold = max(threshold, median * (1 + weight)) # threshold와 가중치 중앙값 중 큰 값을 threshold로 설정

    return light_means > threshold

def classify_traffic_image(image: np.ndarray, light_count: int = 4) -> Tuple[ELabel, Tuple[bool, bool, bool, bool]]:
    # 색을 검사할 샘플 점들, normalized된 이미지에서의 좌표(0~1)
    sample_directions = np.array((
        (0.5, 0.5),
        (0.4, 0.4),
        (0.6, 0.4),
        (0.4, 0.6),
        (0.6, 0.6)
    ))

    height, width = image.shape
    light_width = width / light_count

    # 이미지 크기 기준으로 확장
    sample_directions[:, 0] *= height
    sample_directions[:, 1] *= light_width

    image = cv2.GaussianBlur(image, (15, 15), 0)
    
    sample_points = []
    min_value, max_value = np.inf, 0 # 모든 구역에서의 최소, 최대
    for i in range(light_count):
        start_pos = np.array((0, i * light_width), dtype=np.int32) # 구역의 왼쪽 위, 이미지 좌표 (0, 0)

        sample_pos = np.array(start_pos + sample_directions, dtype=np.int32) # 구역에서 샘플링할 점의 위치
        points = image[sample_pos[:, 0], sample_pos[:, 1]]

        # 모든 점에 대해 최소, 최대값 구하기
        for point in points:
            if point < min_value:
                min_value = point
            if point > max_value:
                max_value = point

        sample_points.append(points)

    normalized_points = []
    for sample_point in sample_points: # 모든 점을 0~1로 normalize
        normalized_points.append((sample_point - min_value) / (max_value - min_value))

    mean_color = np.array([np.mean(section) for section in normalized_points])

    activation_mask = get_active_traffic_lights(mean_color)

    label = ELabel.from_mask(activation_mask)

    return label, tuple(activation_mask)


if __name__ == "__main__":
    image_label_lst = read_image_label("data")

    wrong = []
    for image, label in image_label_lst:
        result_label, mask = classify_traffic_image(image)

        if label != result_label:
            wrong.append((image, label, mask))

    for i, (image, label, mask) in enumerate(wrong):
        plt.subplot(len(wrong), 1, i+1)
        plt.imshow(image, cmap='gray')
        plt.title(f"{ELabel.to_str(label)}{ELabel.to_mask(label)} | {ELabel.to_str(ELabel.from_mask(mask))}{mask}")

        print(f"{ELabel.to_str(label)}{ELabel.to_mask(label)} | {ELabel.to_str(ELabel.from_mask(mask))}{mask}")

    print(f"{len(image_label_lst) - len(wrong)}/{len(image_label_lst)}")

    plt.tight_layout()
    plt.show()