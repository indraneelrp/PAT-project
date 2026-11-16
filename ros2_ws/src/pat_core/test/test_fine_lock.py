
import numpy as np

from pat_core.estimator import DelayAwareKalmanEstimator2D
from pat_core.mocked_hardware import FirstOrderFSM


def rms(values):
    values = np.asarray(values)
    return np.sqrt(np.mean(values ** 2))


def simulate_fine_lock(enabled, seed=42):
    rng = np.random.default_rng(seed)

    dt = 0.001
    steps = 5000

    camera_interval = 33
    camera_delay = 20

    camera_std = 0.00002
    gyro_std = 0.0002

    fsm_limit = 0.002

    # Ground-truth beacon angles (radians)
    true_x = 0.001
    true_y = -0.0006

    # Kalman estimator
    estimator = DelayAwareKalmanEstimator2D(
        history_capacity=300
    )

    for axis in [
        estimator.kf.x_axis,
        estimator.kf.y_axis
    ]:
        axis.R = camera_std ** 2
        axis.P = np.diag([1e-6, 1e-6])
        axis.Q = np.diag([1e-12, 1e-10])

    # Simulated physical steering mirrors
    fsm_x = FirstOrderFSM(tau=0.010)
    fsm_y = FirstOrderFSM(tau=0.010)

    command_x = 0.0
    command_y = 0.0

    # Camera measurements waiting for delivery
    camera_queue = {}

    errors = []

    for k in range(1, steps + 1):
        t = k * dt

        # ----------------------------------
        # 1. Simulate actual platform motion
        # ----------------------------------
        true_gyro_x = (
            0.003 * np.sin(2 * np.pi * 2 * t)
            + 0.001 * np.sin(2 * np.pi * 5 * t)
        )

        true_gyro_y = (
            0.0025 * np.cos(2 * np.pi * 3 * t)
        )

        # Platform rotation moves the apparent beacon
        true_x -= true_gyro_x * dt
        true_y -= true_gyro_y * dt

        # ----------------------------------
        # 2. Simulate IMU measurements
        # ----------------------------------
        measured_gyro_x = (
            true_gyro_x + rng.normal(0, gyro_std)
        )

        measured_gyro_y = (
            true_gyro_y + rng.normal(0, gyro_std)
        )

        # Predict using IMU at 1 kHz
        estimator.predict(
            dt, measured_gyro_x, measured_gyro_y
        )

        # ----------------------------------
        # 3. Simulate delayed camera
        # ----------------------------------
        if k % camera_interval == 0:

            # Camera observes residual pointing error
            residual_x = (
                true_x - fsm_x.position
                + rng.normal(0, camera_std)
            )

            residual_y = (
                true_y - fsm_y.position
                + rng.normal(0, camera_std)
            )

            # Reconstruct LOS measurement using
            # the ACTUAL FSM position at capture time
            observed_x = residual_x + fsm_x.position
            observed_y = residual_y + fsm_y.position

            camera_queue[k + camera_delay] = (
                k, observed_x, observed_y
            )

        # Deliver camera observations after latency
        if k in camera_queue:
            capture_tick, observed_x, observed_y = (
                camera_queue.pop(k)
            )

            estimator.update_camera_at(
                capture_tick,
                observed_x,
                observed_y
            )

        # ----------------------------------
        # 4. Compute FSM commands
        # ----------------------------------
        if enabled:
            estimated_x, estimated_y = estimator.angle

            # Command mirror toward estimated LOS
            command_x = float(np.clip(
                estimated_x, -fsm_limit, fsm_limit
            ))

            command_y = float(np.clip(
                estimated_y, -fsm_limit, fsm_limit
            ))

        # ----------------------------------
        # 5. Simulate physical FSM response
        # ----------------------------------
        fsm_x.step(command_x, dt)
        fsm_y.step(command_y, dt)

        # ----------------------------------
        # 6. Measure ACTUAL pointing error
        # ----------------------------------
        actual_error = np.hypot(
            true_x - fsm_x.position,
            true_y - fsm_y.position
        )

        errors.append(actual_error)

    return np.array(errors)


def test_fine_lock_reduces_pointing_error():
    # Same simulated motion and sensor noise
    uncontrolled = simulate_fine_lock(enabled=False)
    controlled = simulate_fine_lock(enabled=True)

    # Ignore first 500 ms while lock is established
    start = 500

    uncontrolled_rms = rms(uncontrolled[start:])
    controlled_rms = rms(controlled[start:])

    improvement = (
        1.0 - controlled_rms / uncontrolled_rms
    ) * 100

    print(
        f"Uncontrolled RMS: {uncontrolled_rms * 1000:.4f} mrad"
    )
    print(
        f"Controlled RMS:   {controlled_rms * 1000:.4f} mrad"
    )
    print(f"Improvement:      {improvement:.1f}%")

    # Require over 90% RMS error reduction
    assert controlled_rms < 0.1 * uncontrolled_rms

    # Require steady-state fine-pointing accuracy
    assert controlled_rms < 0.0001

