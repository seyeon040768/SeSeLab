import pygame
import numpy as np
from estimate_position import Estimator

# 색상
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

class Car:
    def __init__(self, position: np.ndarray, rotation: np.ndarray, max_speed: float, max_steer: float, acceleration: float, steering: float):
        self.position = position
        self.rotation = rotation

        self.speed = 0
        self.max_speed = max_speed
        self.acceleration = acceleration

        self.steer = 0
        self.max_steer = max_steer
        self.steering = steering

        self.size = (40, 20)
        self.color = (255, 0, 0)

        self.car_surface = pygame.Surface(self.size)
        self.car_surface.fill(self.color)

    def transform(self):
        theta = np.radians(self.steer)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        m_rotation = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])

        self.rotation @= m_rotation

        self.rotation /= np.linalg.norm(self.rotation)
        self.position += self.rotation * self.speed
        

    def update(self):
        keys = pygame.key.get_pressed()
    
        if keys[pygame.K_w]:
            self.speed = self.acceleration
        elif keys[pygame.K_s]:
            self.speed = -self.acceleration
        else:
            self.speed = 0

        if keys[pygame.K_a]:
            self.steer = self.steering * (self.speed / self.max_speed)
        elif keys[pygame.K_d]:
            self.steer = -self.steering * (self.speed / self.max_speed)
        else:
            self.steer = 0


        self.speed = np.clip(self.speed, -self.max_speed, self.max_speed) / max_fps
        self.steer = np.clip(self.steer, -self.max_steer, self.max_steer) / max_fps

        return self.speed, self.steer

    def draw(self):
        angle = np.degrees(np.arctan2(self.rotation[1], self.rotation[0]))

        position = basis_transform(self.position)

        rect_surface = pygame.Surface(self.size, pygame.SRCALPHA)
        pygame.draw.rect(rect_surface, self.color, (0, 0, *self.size))

        # Surface 회전
        rotated_surface = pygame.transform.rotate(rect_surface, angle)

        # 회전된 사각형의 새로운 rect 얻기
        rect = rotated_surface.get_rect()
        rect.center = position

        # 회전된 사각형 그리기
        screen.blit(rotated_surface, rect)


def basis_transform(vec: np.ndarray):
    m_world = np.array([[1, 0], [0, -1]], dtype=np.float32)
    return vec @ m_world + np.array([width / 2, height / 2], dtype=np.float32)

pygame.init()

# 화면 설정
width, height = 800, 600
max_fps = 30
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Top-Down Racing Game")

position = np.array([0, -height / 2 + 100], dtype=np.float32)
rotation = np.array([0.0, 1.0], dtype=np.float32)
car = Car(position, rotation, max_speed=100, max_steer=50, acceleration=100, steering=50)

estimator = Estimator(car.position, car.rotation)

# 시계 설정
clock = pygame.time.Clock()

dots = []

# 메인 게임 루프
running = True
while running:
    screen.fill(WHITE)

    car_speed, car_steer = car.update()
    car.transform()
    car.draw()

    estimated_position, estimated_rotation = estimator.update(car_speed * max_fps, car_steer * max_fps, 1 / max_fps)
    estimated_position_screen = basis_transform(estimated_position)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pos = estimated_position_screen
                dots.append((int(pos[0]), int(pos[1])))

                print(f"real: ({car.position[0]:.2f}, {car.position[1]:.2f})\testimation: ({estimated_position[0]:.2f}, {estimated_position[1]:.2f})\terror: {np.linalg.norm(car.position - estimated_position)}")

    for dot in dots:
        pygame.draw.circle(screen, RED, dot, 5)

    pygame.display.flip()
    clock.tick(max_fps)

pygame.quit()
