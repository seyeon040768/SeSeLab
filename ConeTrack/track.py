import numpy as np
import matplotlib.pyplot as plt
import pygame
import pickle
from typing import Tuple, List, Optional, Union
from numpy.typing import NDArray

class Track:
    """
    트랙 클래스는 트랙의 좌우 경계선을 관리합니다.
    
    Attributes:
        left (NDArray): 트랙 왼쪽 경계선의 좌표점들
        right (NDArray): 트랙 오른쪽 경계선의 좌표점들
        x_range (Tuple[float, float]): 트랙의 x축 범위
        y_range (Tuple[float, float]): 트랙의 y축 범위
        start_pos (Tuple[float, float]): 시작 위치
        start_dir (Tuple[float, float]): 시작 방향
    """
    
    def __init__(self, 
                 left: NDArray, 
                 right: NDArray, 
                 x_range: Tuple[float, float], 
                 y_range: Tuple[float, float],
                 start_pos: Tuple[float, float] = (0, 0),
                 start_dir: Tuple[float, float] = (0, 1)) -> None:
        self.left = left[np.random.permutation(len(left))]
        self.right = right[np.random.permutation(len(right))]
        self.x_range = x_range
        self.y_range = y_range
        self.start_pos = start_pos
        self.start_dir = start_dir

    def visualize(self, 
                 ax: plt.Axes, 
                 path: Optional[NDArray] = None, 
                 margin: int = 20) -> plt.Axes:
        """
        트랙을 시각화합니다.
        
        Args:
            ax: 그래프를 그릴 matplotlib axes
            path: 선택적으로 표시할 경로점들
            margin: 그래프 여백 크기
            
        Returns:
            시각화된 그래프의 axes
        """
        ax.scatter(self.left[:, 0], self.left[:, 1], c='red', label='Left Points', s=10)
        ax.scatter(self.right[:, 0], self.right[:, 1], c='blue', label='Right Points', s=10)
        ax.scatter(0, 0, c='black', label='Origin', s=10)

        if path is not None:
            ax.scatter(path[:, 0], path[:, 1], c='green', label='Path', s=10)
            ax.plot(path[:, 0], path[:, 1], 'g-', alpha=0.5)
        
        ax.set_title('Track')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        ax.set_xlim(self.x_range[0]-margin, self.x_range[1]+margin)
        ax.set_ylim(self.y_range[0]-margin, self.y_range[1]+margin)
        
        ax.set_aspect('equal')
        
        return ax

    def save(self, path: str) -> None:
        """
        트랙 객체를 파일로 저장합니다.
        
        Args:
            path: 저장할 파일 경로
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str) -> 'Track':
        """
        파일에서 트랙 객체를 불러옵니다.
        
        Args:
            path: 불러올 파일 경로
            
        Returns:
            불러온 Track 객체
        """
        with open(path, 'rb') as f:
            track = pickle.load(f)
        return track
    
    @staticmethod
    def make_track(screen_size: Tuple[int, int] = (800, 600)) -> 'Track':
        """
        GUI를 통해 새로운 트랙을 생성합니다.
        
        Args:
            screen_size: 화면 크기 (너비, 높이)
            
        Returns:
            생성된 Track 객체
        """
        pygame.init()

        width = screen_size[0]
        height = screen_size[1]
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Track Generator")

        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        BLUE = (0, 0, 255)
        BLACK = (0, 0, 0)
        GRAY = (128, 128, 128)

        left_points = []
        right_points = []
        origin = None

        running = True
        while running:
            screen.fill(WHITE)

            for x in range(0, width, 100):
                pygame.draw.line(screen, GRAY, (x, 0), (x, height), 1)
            for y in range(0, height, 100):
                pygame.draw.line(screen, GRAY, (0, y), (width, y), 1)
            
            for point in left_points:
                pygame.draw.circle(screen, RED, point, 5)
            for point in right_points:
                pygame.draw.circle(screen, BLUE, point, 5)
            
            if origin != None:
                pygame.draw.circle(screen, BLACK, origin, 5)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if event.button == 1:
                        left_points.append(pos)
                    elif event.button == 3:
                        right_points.append(pos)
                    elif event.button == 2:
                        origin = pos
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q and origin is not None:
                    left_array = np.array(left_points)
                    left_array[:, 1] = screen_size[1] - left_array[:, 1]

                    right_array = np.array(right_points)
                    right_array[:, 1] = screen_size[1] - right_array[:, 1]

                    origin = (origin[0], screen_size[1] - origin[1])

                    running = False
            
            pygame.display.flip()

        pygame.quit()

        left_points, right_points = left_array, right_array

        x_range = (-origin[0], screen_size[0] - origin[0])
        y_range = (-origin[1], screen_size[1] - origin[1])

        left_points[:, 0] -= origin[0]
        right_points[:, 0] -= origin[0]
        left_points[:, 1] -= origin[1]
        right_points[:, 1] -= origin[1]

        track = Track(left_points, right_points, x_range, y_range)
        
        return track
    

def view_tracks(track_files: List[str], 
                make_path_func: Optional[callable] = None) -> None:
    """
    여러 트랙을 동시에 시각화합니다.
    
    Args:
        track_files: 트랙 파일 경로 목록
        make_path_func: 선택적으로 경로를 생성하는 함수
    """
    n_tracks = len(track_files)
    if n_tracks == 1:
        n_cols = 1
        n_rows = 1
    elif n_tracks == 2:
        n_cols = 2
        n_rows = 1
    elif n_tracks == 3 or n_tracks == 4:
        n_cols = 2
        n_rows = 2
    else:
        n_cols = 3
        n_rows = (n_tracks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(5*n_cols, 5*n_rows),
                            gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = np.array(axes).flatten()

    for i, track_file in enumerate(track_files):
        if track_file == "custom":
            track = Track.make_track()
        else:
            with open(track_file, 'rb') as f:
                track = pickle.load(f)

        path = None
        if make_path_func is not None:
            path = make_path_func(track)
        track.visualize(axes[i], path=path)

        axes[i].set_title(track_file.split('/')[-1])

    for i in range(n_tracks, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(pad=1.5)
    plt.show()