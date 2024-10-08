import numpy as np

class Estimator:
    def __init__(self, position: np.ndarray, rotation: np.ndarray, max_speed: float = 100, max_steer: float = 50, max_speed_skidding: float = 1):
        self.position = position
        self.rotation = rotation

        self.previous_speed = 0
        self.skidding_coef = max_speed_skidding / max_speed**2

    def update(self, speed: float, steer: float, deltatime: float):
        speed *= deltatime
        steer *= deltatime

        theta = np.radians(steer)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        m_rotation = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])

        self.rotation @= m_rotation

        self.rotation /= np.linalg.norm(self.rotation)
        self.position += self.rotation * speed

        self.previous_speed = speed

        return self.position, self.rotation
