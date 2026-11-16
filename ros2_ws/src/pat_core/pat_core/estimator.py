'''
Kalman filter and estimator
'''
import numpy as np


class KalmanEstimator1D:
    def __init__(self):
        # State vector [angular error, angular velocity]
        self.x = np.zeros(2)

        # Uncertainty matrix: uncertainty of the state represented by Variance and covariance
        self.P = np.eye(2) # 2 x 2 identity matrix -- 1 variance, 0 covariance

        # Additive noise
        self.Q = np.diag([1e-8, 1e-6])

        # Camera measurement noise variance
        self.R = 1e-8

        # Matrix to select the error only
        self.H = np.array([[1.0, 0.0]])

    # predict step- update step- based on known IMU behaviour
    def predict(self, dt, gyro=0.0):
        # for a certain axis, the state is 2 values
        # column matrix ( theta, omega )
        # theta is this state's error value
        # omega is this state's curr angular velocity

        # F is the TRANSITION matrix. Multiply F with the prev state x to get the new state column(theta_new, omega_new)
        # We need to update the prediction for next state,
        # and update the prediction for the next uncertainty (matrix P)

        # State
        F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])
        B = np.array([-dt, 0.0]) # include accounting for gyro
        self.x = F @ self.x + B * gyro

        # Uncertainty
        self.P = F @ self.P @ F.T + self.Q


    # Update based on a camera observation, adjust the state and uncertainty
    # The kalman update.
    def update_camera(self, measured_error):
        ## -- Camera measurement --
        z = np.array([measured_error])

        ## -- Kalman gain --
        # P @ H.T @ [inv( H @P @H.T + R )]     <- H is just a selector matrix.
        # Equation arises from setting the values of K that will minimise 
        # variances P and R (where P and R have H applied already). 
        # P is prediction's variance and R is reported measurement's variance
        intermediate = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(intermediate)

        ## -- Correct state --
        # innovation residual first, then apply K to resudual + old state for full correction
        y = z - self.H @ self.x
        self.x = self.x + K @ y

        ## -- Correct uncertainty --
        I = np.eye(2)
        A = I - K @ self.H
        self.P = A @ self.P @ A.T + (K * self.R) @ K.T


class KalmanEstimator2D:
    def __init__(self):
        # Separate estimator for each angular axis
        self.x_axis = KalmanEstimator1D()
        self.y_axis = KalmanEstimator1D()

    def predict(self, dt, gyro_x, gyro_y):
        # Predict both axes from IMU measurements
        self.x_axis.predict(dt, gyro_x)
        self.y_axis.predict(dt, gyro_y)

    def update_camera(self, measured_x, measured_y):
        # Correct both axes from camera measurements
        self.x_axis.update_camera(measured_x)
        self.y_axis.update_camera(measured_y)

    @property
    def angle(self):
        return (
            self.x_axis.x[0],
            self.y_axis.x[0]
        )

    @property
    def drift(self):
        return (
            self.x_axis.x[1],
            self.y_axis.x[1]
        )


class DelayAwareKalmanEstimator2D:
    def __init__(self, history_capacity=300):
        self.kf = KalmanEstimator2D()
        self.tick = 0
        self.history_capacity = history_capacity
        self.last_camera_tick = -1

        # Store initial state at tick 0
        self.history = [self._snapshot(0, 0, 0, 0)]

    def _snapshot(self, tick, dt, gyro_x, gyro_y):
        return {
            "tick": tick,
            "dt": dt,
            "gyro_x": gyro_x,
            "gyro_y": gyro_y,

            # Save copies, not references!
            "x": self.kf.x_axis.x.copy(),
            "Px": self.kf.x_axis.P.copy(),
            "y": self.kf.y_axis.x.copy(),
            "Py": self.kf.y_axis.P.copy(),
        }

    def _restore(self, snapshot):
        self.kf.x_axis.x = snapshot["x"].copy()
        self.kf.x_axis.P = snapshot["Px"].copy()
        self.kf.y_axis.x = snapshot["y"].copy()
        self.kf.y_axis.P = snapshot["Py"].copy()

    def predict(self, dt, gyro_x, gyro_y):
        self.kf.predict(dt, gyro_x, gyro_y)

        self.tick += 1

        self.history.append(
            self._snapshot(self.tick, dt, gyro_x, gyro_y)
        )

        # Keep memory bounded
        if len(self.history) > self.history_capacity:
            self.history.pop(0)

    def update_camera_at(self, capture_tick, measured_x, measured_y):
        # This simplified version requires a retained IMU tick.
        # Camera captures must also arrive in capture-time order.
        if capture_tick <= self.last_camera_tick:
            raise ValueError("Out-of-order camera capture")

        index = next(
            (i for i, s in enumerate(self.history)
             if s["tick"] == capture_tick),
            None
        )

        if index is None:
            raise ValueError("Camera capture outside stored history")

        # 1. Restore estimator at image capture time
        self._restore(self.history[index])

        # 2. Correct that historical state
        self.kf.update_camera(measured_x, measured_y)

        # 3. Save corrected historical state
        original = self.history[index]
        self.history[index] = self._snapshot(
            original["tick"],
            original["dt"],
            original["gyro_x"],
            original["gyro_y"],
        )

        # 4. Replay all subsequent IMU predictions
        for j in range(index + 1, len(self.history)):
            old = self.history[j]

            self.kf.predict(
                old["dt"],
                old["gyro_x"],
                old["gyro_y"],
            )

            self.history[j] = self._snapshot(
                old["tick"],
                old["dt"],
                old["gyro_x"],
                old["gyro_y"],
            )

        self.last_camera_tick = capture_tick

    @property
    def angle(self):
        return self.kf.angle
