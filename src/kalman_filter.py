
import numpy as np


class KalmanFilter3D:

    def __init__(self, dt=1.0):

        # State transition matrix
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ], dtype=float)

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=float)

        # Process noise
        self.Q = np.eye(6) * 0.1

        # Empirically estimated measurement noise covariance
        self.R = np.eye(3) * 0.5

        # Initial covariance
        self.P = np.eye(6) * 100

        # Initial state
        self.x = np.zeros((6, 1))

    def initialize(self, position, velocity=None):

        self.x[:3] = np.asarray(position).reshape(3, 1)

        if velocity is not None:
            self.x[3:] = np.asarray(velocity).reshape(3, 1)

    def predict(self):

        # Predict state
        self.x = self.F @ self.x

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x[:3].flatten()

    def update(self, measurement):

        z = np.asarray(measurement).reshape(3, 1)

        # Measurement residual
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman Gain
        K = np.linalg.solve(
            S.T,
            (self.P @ self.H.T).T
        ).T

        # Correct state
        self.x = self.x + K @ y

        # Joseph covariance update
        I = np.eye(6)
        A = I - K @ self.H

        self.P = (
            A @ self.P @ A.T
            + K @ self.R @ K.T
        )

        return self.x[:3].flatten()


# Test the Kalman Filter
if __name__ == "__main__":

    kf = KalmanFilter3D(dt=1.0)

    kf.initialize(
        position=[0, 0, 0],
        velocity=[1, 0, 0]
    )

    print("Initial State:")
    print(kf.x.flatten())

    predicted = kf.predict()

    print("\nPredicted Position:")
    print(predicted)

    measurement = [1.2, 0.1, 0.0]

    updated = kf.update(measurement)

    print("\nMeasured Position:")
    print(measurement)

    print("\nUpdated Position:")
    print(updated)

    print("\nUpdated Velocity:")
    print(kf.x[3:].flatten())

    print("\nKalman Filter Test Completed!")
