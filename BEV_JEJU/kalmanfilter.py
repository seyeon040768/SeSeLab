import numpy as np

class KalmanFilter2D:
    def __init__(self, dt, init_state, init_cov, process_var, measure_var):
        self.x = init_state
        self.P = init_cov

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        q = process_var
        self.Q = np.array([
            [0.25 * dt**4 * q, 0, 0.5 * dt**3 * q, 0],
            [0, 0.25 * dt**4 * q, 0, 0.5 * dt**3 * q],
            [0.5 * dt**3 * q, 0, dt**2 * q, 0],
            [0, 0.5 * dt**3 * q, 0, dt**2 * q]
        ])

        self.H = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.R = np.diag([measure_var, measure_var])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P

if __name__ == "__main__":
    dt = 1 / 30.0

    init_state = np.array([0.0, 0.0, 0.0, 0.0])[:, np.newaxis]
    init_cov = np.diag([1.0, 1.0, 1.0, 1.0])
    process_var = 0.1
    measure_var = 0.1

    kf = KalmanFilter2D(dt, init_state, init_cov, process_var, measure_var)

    