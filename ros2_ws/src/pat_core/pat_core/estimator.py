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