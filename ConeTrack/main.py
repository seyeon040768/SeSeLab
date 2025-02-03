import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import glob

from track import Track, view_tracks

def make_path(track: Track) -> np.ndarray:
    """
    경로를 생성하는 함수

    Args:
        track (Track): 경로

    Returns:
        NDArray: n*2 형태의 경로 점들
    """
    start_pos = (0, 0) # 필요 시 사용, 시작점은 언제나 (0, 0)

    # TODO: 함수 완성하기!
    # 시작점부터 끝점까지 10개의 점을 균등하게 생성
    start_pos = np.array(start_pos, dtype=np.float64)
    start_dir = np.array(track.start_dir, dtype=np.float64)

    left_points = track.left
    right_points = track.right

    # 시작점에서 트랙을 가리키는 방향 벡터
    left_dir = left_points - start_pos
    right_dir = right_points - start_pos

    # 시작점에서 트랙과의 거리
    left_dist = np.linalg.norm(left_dir, axis=1)
    right_dist = np.linalg.norm(right_dir, axis=1)

    # 단위 벡터로 만들기
    left_dir = left_dir / left_dist[:, np.newaxis]
    right_dir = right_dir / right_dist[:, np.newaxis]

    # 시작 방향과 트랙 사이의 방향 차이 계산
    left_dot = left_dir @ start_dir.T
    right_dot = right_dir @ start_dir.T

    left_dist_sorted = np.argsort(left_dist)
    right_dist_sorted = np.argsort(right_dist)

    left_ordered = [left_points[left_dist_sorted[0]]]
    right_ordered = [right_points[right_dist_sorted[0]]]

    left_dist_ordered = [left_dist[left_dist_sorted[0]]]
    right_dist_ordered = [right_dist[right_dist_sorted[0]]]

    # 가려는 방향에 있는 점 중 가장 가까운 점 찾기
    i = 1
    while left_dot[left_dist_sorted[i]] < 0:
        i += 1
    left_ordered.append(left_points[left_dist_sorted[i]])
    left_dist_ordered.append(left_dist[left_dist_sorted[i]])

    # 남아있는 점에서 재귀적으로 가까운 점 찾기
    remaining_indices = set(range(len(left_points))) - {left_dist_sorted[0], left_dist_sorted[i]}
    while remaining_indices:
        remaining_indices_lst = list(remaining_indices)
        distances = np.linalg.norm(left_points[remaining_indices_lst] - left_ordered[-1], axis=1)
        closest_idx = remaining_indices_lst[np.argmin(distances)]
        
        left_ordered.append(left_points[closest_idx])
        left_dist_ordered.append(left_dist[closest_idx])
        remaining_indices.remove(closest_idx)

    i = 1
    while right_dot[right_dist_sorted[i]] < 0:
        i += 1
    right_ordered.append(right_points[right_dist_sorted[i]])
    right_dist_ordered.append(right_dist[right_dist_sorted[i]])

    remaining_indices = set(range(len(right_points))) - {right_dist_sorted[0], right_dist_sorted[i]}
    while remaining_indices:
        remaining_indices_lst = list(remaining_indices)
        distances = np.linalg.norm(right_points[remaining_indices_lst] - right_ordered[-1], axis=1)
        closest_idx = remaining_indices_lst[np.argmin(distances)]
        
        right_ordered.append(right_points[closest_idx])
        right_dist_ordered.append(right_dist[closest_idx])
        remaining_indices.remove(closest_idx)

    left_idx, right_idx = 0, 0
    result = [(left_points[left_idx] + right_points[right_idx]) / 2]
    while left_idx + 1 < len(left_points) and right_idx + 1 < len(right_points):
        pivot = result[-1]
        
        left_pivot_dist = np.linalg.norm(left_points[left_idx + 1] - pivot)
        right_pivot_dist = np.linalg.norm(right_points[right_idx + 1] - pivot)

        diff = left_pivot_dist / right_pivot_dist
        if diff < 0.8:
            left_idx += 1
        elif diff > 1 / 0.8:
            right_idx += 1
        else:
            left_idx += 1
            right_idx += 1

        result.append((left_points[left_idx] + right_points[right_idx]) / 2)
    
    result = np.array(result)

    # spline
    t = np.linspace(0, 1, len(result))
    cs = CubicSpline(t, result)
    
    # 스플라인의 총 길이 계산
    t_eval = np.linspace(0, 1, 1000)
    points = cs(t_eval)
    diffs = np.diff(points, axis=0)
    lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative_lengths = np.concatenate(([0], np.cumsum(lengths)))
    total_length = cumulative_lengths[-1]
    
    # 동일한 간격으로 점 생성
    desired_distances = np.linspace(0, total_length, len(result))
    t_uniform = np.interp(desired_distances, cumulative_lengths, t_eval)
    t_uniform = np.append(t_uniform, 1.0)
    result = cs(t_uniform)
    
    return result

    

if __name__ == "__main__":
    level1 = ("tracks/straight.trk", "tracks/curved.trk")
    level2 = ("tracks/straight2.trk", "tracks/curved2.trk", "tracks/perpen.trk")
    level3 = ("tracks/n.trk", "tracks/s.trk", "tracks/circle.trk")
    all = glob.glob("tracks/*.trk")
    custom = ("custom", )

    # view_tracks(all, make_path_func=None)
    view_tracks(all, make_path_func=make_path) # change level
