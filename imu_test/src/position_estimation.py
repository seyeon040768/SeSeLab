import numpy as np

class PositionEstimator:
    def __init__(self):
        self.position = np.array([0, 0, 0], dtype=np.float64)
        self.velocity = np.array([0, 0, 0], dtype=np.float64)
        self.prev_time = 0
        self.prepare_count = 100
        self.error_mean = np.array([0, 0, 0], dtype=np.float64)

    def update(self, timestamp, accel):
        if self.prepare_count > 0:
            self.prev_time = timestamp

            self.error_mean += accel

            if self.prepare_count == 1:
                self.error_mean /= 100
            
            self.prepare_count -= 1
            return
        
        delta_time = timestamp - self.prev_time

        accel -= self.error_mean
        accel[np.abs(accel) < 0.01] = 0.0

        new_velocity = self.velocity + accel * delta_time
        mean_velocity = (self.velocity + new_velocity) / 2

        self.position += mean_velocity * delta_time

        self.velocity = new_velocity
        self.prev_time = timestamp

        print("deltatime:", delta_time)
        print("accel:", accel)

        # TODO: use angle, angle_accel and test ekf
