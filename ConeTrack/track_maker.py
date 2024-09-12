import pygame
import numpy as np
from path_generator import generate_path

screen_size = (800, 800)

# Pygame 초기화
pygame.init()

# 화면 크기 설정
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("좌표 기록 프로그램")

# 색상 정의
BLUE = (0, 0, 255)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# 좌표 리스트
left_click_points = []  # 좌클릭 좌표
right_click_points = []  # 우클릭 좌표

# 게임 루프
running = True
while running:
    # 이벤트 처리
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # 좌클릭
                pos = pygame.mouse.get_pos()
                left_click_points.append(pos)
            elif event.button == 3:  # 우클릭
                pos = pygame.mouse.get_pos()
                right_click_points.append(pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # q 키를 누르면 종료
                running = False
            if event.key == pygame.K_c:
                left_click_points = []
                right_click_points = []

    # 화면 업데이트
    screen.fill(WHITE)  # 배경을 흰색으로 채움    

    # 좌클릭 좌표에 파란색 점 그리기
    for point in left_click_points:
        pygame.draw.circle(screen, BLUE, point, 5)

    # 우클릭 좌표에 빨간색 점 그리기
    for point in right_click_points:
        pygame.draw.circle(screen, RED, point, 5)

    
    pygame.draw.circle(screen, (0, 0, 0), (10, screen_size[1] - 10), 5)

    # 화면 새로고침
    pygame.display.flip()



# Pygame 종료
pygame.quit()


left_click_points = np.array(left_click_points, dtype=np.float64)
right_click_points = np.array(right_click_points, dtype=np.float64)

left_click_points[:, 1] = screen_size[1] - left_click_points[:, 1]
right_click_points[:, 1] = screen_size[1] - right_click_points[:, 1]

np.random.shuffle(left_click_points)
np.random.shuffle(right_click_points)

# 기록된 좌표 출력
print("좌클릭 좌표 리스트:", left_click_points)
print("우클릭 좌표 리스트:", right_click_points)

generate_path(left_click_points, right_click_points, b_visualize=True)