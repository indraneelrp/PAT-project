import numpy as np
from pat_core.estimator import KalmanEstimator1D, KalmanEstimator2D, DelayAwareKalmanEstimator2D


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



# -------------------------------------
# KalmanEstimator2D Tests
# -------------------------------------
def test_2d_gyro_prediction():
    kf = KalmanEstimator2D()

    # Initial angular positions
    kf.x_axis.x = np.array([0.001, 0.0])
    kf.y_axis.x = np.array([-0.0005, 0.0])

    dt = 0.01

    # Platform rotation measured on two axes
    gyro_x = 0.02
    gyro_y = -0.01

    kf.predict(dt, gyro_x, gyro_y)

    # X: 0.001 - 0.02 * 0.01 = 0.0008
    assert np.isclose(kf.angle[0], 0.0008)

    # Y: -0.0005 - (-0.01) * 0.01 = -0.0004
    assert np.isclose(kf.angle[1], -0.0004)



def test_2d_camera_update():
    kf = KalmanEstimator2D()

    # Initial estimates are both zero
    assert kf.angle == (0.0, 0.0)

    # Camera detects beacon at:
    # X = +1 mrad, Y = -0.5 mrad
    kf.update_camera(
        measured_x=0.001,
        measured_y=-0.0005
    )

    # X should move toward +1 mrad
    assert kf.angle[0] > 0.0
    assert np.isclose(kf.angle[0], 0.001, atol=1e-5)

    # Y should move toward -0.5 mrad
    assert kf.angle[1] < 0.0
    assert np.isclose(kf.angle[1], -0.0005, atol=1e-5)

    # Both axes should become more certain
    assert kf.x_axis.P[0, 0] < 1.0
    assert kf.y_axis.P[0, 0] < 1.0


# fuller test with different rotation around x and y axes
# and trying to stabilise to a target position for both axes
def test_2d_imu_camera_fusion():
    rng = np.random.default_rng(42)

    kf = KalmanEstimator2D()

    dt = 0.001
    steps = 2000

    # True beacon angles
    true_x = 0.001
    true_y = -0.0005

    camera_std = 0.0001
    gyro_std = 0.0002

    # Configure both estimators
    for axis in [kf.x_axis, kf.y_axis]:
        axis.R = camera_std ** 2
        axis.P = np.diag([1e-6, 1e-6])
        axis.Q = np.diag([1e-12, 1e-10])

    errors = []

    for k in range(steps):
        t = k * dt

        # Simulated rotation on two axes
        true_gyro_x = 0.005 * np.sin(2 * np.pi * 2 * t)
        true_gyro_y = 0.003 * np.cos(2 * np.pi * 3 * t)

        # Actual beacon motion in camera coordinates
        true_x -= true_gyro_x * dt
        true_y -= true_gyro_y * dt

        # Noisy IMU readings
        measured_gyro_x = (
            true_gyro_x + rng.normal(0, gyro_std)
        )
        measured_gyro_y = (
            true_gyro_y + rng.normal(0, gyro_std)
        )

        # High-rate prediction
        kf.predict(
            dt,
            measured_gyro_x,
            measured_gyro_y
        )

        # Camera measurement every 33 ms
        if k % 33 == 0:
            measured_x = (
                true_x + rng.normal(0, camera_std)
            )
            measured_y = (
                true_y + rng.normal(0, camera_std)
            )

            kf.update_camera(measured_x, measured_y)

        # Compute 2D angular estimation error
        estimate_x, estimate_y = kf.angle

        error = np.hypot(
            estimate_x - true_x,
            estimate_y - true_y
        )

        errors.append(error)

    # Ignore first 200 ms
    errors = np.array(errors[200:])

    rmse = np.sqrt(np.mean(errors ** 2))

    print(f"2D fusion RMSE: {rmse:.8f} rad")

    assert rmse < 0.00015



def test_delayed_camera_replay():
    dt = 0.001
    gyro_x = 0.02
    gyro_y = -0.01

    # True initial beacon angles
    initial_x = 0.001
    initial_y = -0.0005

    delayed = DelayAwareKalmanEstimator2D()

    # Naive estimator applies old camera data at current time
    naive = KalmanEstimator2D()

    for k in range(1, 121):
        delayed.predict(dt, gyro_x, gyro_y)
        naive.predict(dt, gyro_x, gyro_y)

    # Camera captured at tick 100 (100 ms)
    capture_tick = 100

    # The camera's actual observation at capture time
    camera_x = initial_x - gyro_x * 0.100
    camera_y = initial_y - gyro_y * 0.100

    # Both estimators receive it at tick 120 (120 ms)
    delayed.update_camera_at(
        capture_tick, camera_x, camera_y
    )

    naive.update_camera(camera_x, camera_y)

    # Actual angles at the CURRENT time, 120 ms
    true_x = initial_x - gyro_x * 0.120
    true_y = initial_y - gyro_y * 0.120

    delayed_error = np.hypot(
        delayed.angle[0] - true_x,
        delayed.angle[1] - true_y
    )

    naive_error = np.hypot(
        naive.angle[0] - true_x,
        naive.angle[1] - true_y
    )

    print(f"Naive error:   {naive_error:.8f} rad")
    print(f"Delayed error: {delayed_error:.8f} rad")

    assert delayed_error < 0.1 * naive_error



def test_repeated_delayed_camera_fusion():
    rng = np.random.default_rng(42)

    dt = 0.001
    steps = 3000
    camera_interval = 33
    camera_delay = 20

    camera_std = 0.00002
    gyro_std = 0.0002

    # Two estimators receiving identical observations
    naive = KalmanEstimator2D()
    delayed = DelayAwareKalmanEstimator2D(
        history_capacity=300
    )

    # Configure the filters identically
    for kf in [naive, delayed.kf]:
        for axis in [kf.x_axis, kf.y_axis]:
            axis.R = camera_std ** 2
            axis.P = np.diag([1e-6, 1e-6])
            axis.Q = np.diag([1e-12, 1e-10])

    # True beacon angle
    true_x = 0.001
    true_y = -0.0005

    # Pending camera measurements
    # key = delivery tick
    # value = (capture tick, measured x, measured y)
    camera_queue = {}

    naive_errors = []
    delayed_errors = []

    for k in range(1, steps + 1):
        t = k * dt

        # 1. Simulate platform vibration / rotation
        true_gyro_x = (
            0.03 * np.sin(2 * np.pi * 2 * t)
            + 0.012 * np.sin(2 * np.pi * 5 * t)
        )

        true_gyro_y = (
            0.02 * np.cos(2 * np.pi * 3 * t)
        )

        # 2. Update ground-truth beacon angles
        true_x -= true_gyro_x * dt
        true_y -= true_gyro_y * dt

        # 3. Generate noisy IMU measurements
        measured_gyro_x = (
            true_gyro_x + rng.normal(0, gyro_std)
        )
        measured_gyro_y = (
            true_gyro_y + rng.normal(0, gyro_std)
        )

        # 4. Predict both estimators every 1 ms
        naive.predict(
            dt, measured_gyro_x, measured_gyro_y
        )

        delayed.predict(
            dt, measured_gyro_x, measured_gyro_y
        )

        # 5. Camera captures an image every 33 ms
        if k % camera_interval == 0:
            measured_x = (
                true_x + rng.normal(0, camera_std)
            )
            measured_y = (
                true_y + rng.normal(0, camera_std)
            )

            delivery_tick = k + camera_delay

            camera_queue[delivery_tick] = (
                k, measured_x, measured_y
            )

        # 6. Deliver previously captured images
        if k in camera_queue:
            capture_tick, measured_x, measured_y = (
                camera_queue.pop(k)
            )

            # Naive: apply old measurement NOW
            naive.update_camera(
                measured_x, measured_y
            )

            # Correct: update historical state,
            # then replay IMU measurements
            delayed.update_camera_at(
                capture_tick, measured_x, measured_y
            )

        # 7. Measure 2D estimation error
        naive_error = np.hypot(
            naive.angle[0] - true_x,
            naive.angle[1] - true_y
        )

        delayed_error = np.hypot(
            delayed.angle[0] - true_x,
            delayed.angle[1] - true_y
        )

        naive_errors.append(naive_error)
        delayed_errors.append(delayed_error)

    # Ignore first 300 ms of initial convergence
    naive_errors = np.array(naive_errors[300:])
    delayed_errors = np.array(delayed_errors[300:])

    naive_rmse = np.sqrt(np.mean(naive_errors ** 2))
    delayed_rmse = np.sqrt(np.mean(delayed_errors ** 2))

    improvement = (
        1.0 - delayed_rmse / naive_rmse
    ) * 100

    print(f"Naive RMSE:   {naive_rmse:.8f} rad")
    print(f"Replay RMSE:  {delayed_rmse:.8f} rad")
    print(f"Improvement: {improvement:.1f}%")

    assert delayed_rmse < naive_rmse
    assert delayed_rmse < 0.0001
