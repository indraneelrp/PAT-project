import numpy as np
from pat_core.estimator import KalmanEstimator1D


def test_camera_update_reduces_uncertainty():
    kf = KalmanEstimator1D()

    before = kf.P[0, 0]

    kf.update_camera(0.001)

    after = kf.P[0, 0]

    assert after < before

    # Estimated angular error should approach 1 mrad
    assert np.isclose(kf.x[0], 0.001, atol=1e-5)


# test that when initial state 0, but there is noise, the implementation should converge once again to 
# small values after many steps
def test_noisy_camera_convergence():
    rng = np.random.default_rng(42)

    kf = KalmanEstimator1D()

    # Camera noise: standard deviation = 0.1 mrad
    noise_std = 0.0001
    kf.R = noise_std ** 2

    # Small process noise: approximately stationary target
    kf.Q = np.diag([1e-12, 1e-10])

    # Initial state uncertainty
    kf.P = np.diag([1e-6, 1e-6])

    true_angle = 0.001  # 1 mrad
    dt = 1.0 / 30.0     # Camera at 30 Hz

    measurements = []
    estimates = []

    for k in range(300):
        # Predict state forward
        kf.predict(dt)

        # Simulate noisy camera measurement
        measurement = true_angle + rng.normal(
            0.0, noise_std
        )

        # Correct using camera
        kf.update_camera(measurement)

        measurements.append(measurement)
        estimates.append(kf.x[0])

    # Ignore initial convergence period
    measurements = np.array(measurements[50:])
    estimates = np.array(estimates[50:])

    camera_rmse = np.sqrt(
        np.mean((measurements - true_angle) ** 2)
    )

    kalman_rmse = np.sqrt(
        np.mean((estimates - true_angle) ** 2)
    )

    print(f"Camera RMSE: {camera_rmse:.8f}")
    print(f"Kalman RMSE: {kalman_rmse:.8f}")

    assert kalman_rmse < 0.5 * camera_rmse


# based on some initial state x and a dt and a gyro, check if the implementation does a 
# correct next prediction
def test_gyro_prediction():
    kf = KalmanEstimator1D()

    # Initial LOS angle = 1 mrad
    # Zero residual drift
    kf.x = np.array([0.001, 0.0])

    dt = 0.01
    gyro = 0.02  # 20 mrad/s

    kf.predict(dt, gyro)

    # Expected angle:
    # 0.001 - 0.02 * 0.01 = 0.0008
    assert np.isclose(kf.x[0], 0.0008)
    assert np.isclose(kf.x[1], 0.0)


# check now whether based on some movement pattern && INCLDING camera observation whether
# the prediction can stabilise to good values
def test_imu_camera_fusion():
    rng = np.random.default_rng(42)
    kf = KalmanEstimator1D()

    dt = 0.001
    steps = 2000

    # Camera and IMU noise
    camera_std = 0.0001
    gyro_std = 0.0002

    kf.R = camera_std ** 2
    kf.P = np.diag([1e-6, 1e-6])
    kf.Q = np.diag([1e-12, 1e-10])

    # True beacon angle relative to platform
    true_angle = 0.001

    estimates = []
    truths = []
    measurements = []

    for k in range(steps):
        t = k * dt

        # Simulated platform angular velocity
        true_gyro = 0.005 * np.sin(2 * np.pi * 2 * t)

        # Actual platform movement changes beacon angle
        true_angle -= true_gyro * dt

        # Noisy gyroscope measurement
        measured_gyro = (
            true_gyro + rng.normal(0, gyro_std)
        )

        # Predict using IMU
        kf.predict(dt, measured_gyro)

        # Camera measures every ~33 ms
        if k % 33 == 0:
            camera_measurement = (
                true_angle + rng.normal(0, camera_std)
            )

            kf.update_camera(camera_measurement)
            measurements.append(camera_measurement)

        estimates.append(kf.x[0])
        truths.append(true_angle)

    estimates = np.array(estimates)
    truths = np.array(truths)

    rmse = np.sqrt(np.mean(
        (estimates[200:] - truths[200:]) ** 2
    ))

    print(f"Fusion RMSE: {rmse:.8f} rad")

    assert rmse < 0.0001


# test camera-only vs with kalman camera update
def test_camera_only_vs_fusion():
    rng = np.random.default_rng(42)
    kf = KalmanEstimator1D()

    dt = 0.001
    steps = 2000

    # Sensor noise
    camera_std = 0.0001
    gyro_std = 0.0002

    # Kalman configuration
    kf.R = camera_std ** 2
    kf.P = np.diag([1e-6, 1e-6])
    kf.Q = np.diag([1e-12, 1e-10])

    # Actual beacon angle
    true_angle = 0.001

    # Camera-only tracking
    camera_only_angle = 0.0

    # Results for comparison
    truths = []
    fusion_estimates = []
    camera_only_estimates = []

    for k in range(steps):
        t = k * dt

        # 1. Simulate actual platform motion
        true_gyro = 0.005 * np.sin(2 * np.pi * 2 * t)
        true_angle -= true_gyro * dt

        # 2. Generate noisy IMU measurement
        measured_gyro = (
            true_gyro + rng.normal(0, gyro_std)
        )

        # 3. Kalman prediction at 1 kHz
        kf.predict(dt, measured_gyro)

        # 4. Camera measurement every 33 ms
        if k % 33 == 0:
            camera_measurement = (
                true_angle + rng.normal(0, camera_std)
            )

            # Kalman uses the measurement
            kf.update_camera(camera_measurement)

            # Camera-only tracker stores the SAME reading
            camera_only_angle = camera_measurement

        # 5. Record both estimates at every millisecond
        truths.append(true_angle)
        fusion_estimates.append(kf.x[0])
        camera_only_estimates.append(camera_only_angle)

    # Convert to numpy arrays
    truths = np.array(truths)
    fusion_estimates = np.array(fusion_estimates)
    camera_only_estimates = np.array(camera_only_estimates)

    # Ignore first 200 ms for initial convergence
    start = 200

    fusion_rmse = np.sqrt(np.mean(
        (fusion_estimates[start:] - truths[start:]) ** 2
    ))

    camera_only_rmse = np.sqrt(np.mean(
        (camera_only_estimates[start:] - truths[start:]) ** 2
    ))

    print(f"Camera-only RMSE: {camera_only_rmse:.8f} rad")
    print(f"Fusion RMSE:      {fusion_rmse:.8f} rad")

    # Original correctness requirement
    assert fusion_rmse < 0.0001

    # Check whether fusion outperforms this simple baseline
    assert fusion_rmse < camera_only_rmse
