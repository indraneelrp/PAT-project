"""ROS-free validation: same plant, estimator, controller as ROS nodes.

Run `python -m pat_core.offline_validation` with PYTHONPATH set to package root.
The demo compares actuator open-loop, camera-only, naive Kalman, delay-aware
Kalman. Each run uses the SAME seed and simulated platform disturbance.
"""
import math
import numpy as np
from .controller import FSMController
from .estimator import KalmanEstimator2D, DelayAwareKalmanEstimator2D
from .simulator_core import PATSimulator


def run_episode(mode, steps=2000, seed=42, warmup=200, **plant_kwargs):
    if mode not in ('off', 'camera_only', 'naive', 'replay'):
        raise ValueError('Invalid episode mode')
    sim = PATSimulator(seed=seed, **plant_kwargs)
    naive = KalmanEstimator2D()
    replay = DelayAwareKalmanEstimator2D(history_capacity=500)
    for kf in (naive, replay.kf):
        for axis in (kf.x_axis, kf.y_axis):
            axis.R = sim.camera_std ** 2
            axis.P = np.diag([1e-6, 1e-6])
            axis.Q = np.diag([1e-11, 1e-9])
    ctrl_x = FSMController(limit=sim.fsm_limit)
    ctrl_y = FSMController(limit=sim.fsm_limit)
    cmd_x = cmd_y = 0.0
    cam_x = cam_y = 0.0
    recording = []

    for _ in range(steps):
        s = sim.step(cmd_x, cmd_y)
        naive.predict(s.dt, s.gyro_x, s.gyro_y)
        replay.predict(s.dt, s.gyro_x, s.gyro_y)
        for obs in s.camera:
            # The camera sees residual, the KF tracks target LOS.
            theta_x_observed = obs.error_x + obs.fsm_x_at_capture
            theta_y_observed = obs.error_y + obs.fsm_y_at_capture
            cam_x, cam_y = theta_x_observed, theta_y_observed
            naive.update_camera(theta_x_observed, theta_y_observed)
            replay.update_camera_at(obs.capture_tick, theta_x_observed, theta_y_observed)
        if mode == 'off':
            cmd_x = cmd_y = 0.0
            estimate = (0.0, 0.0)
        elif mode == 'camera_only':
            estimate = (cam_x, cam_y)
        elif mode == 'naive':
            estimate = naive.angle
        else:
            estimate = replay.angle
        if mode != 'off':
            cmd_x = ctrl_x.compute_command(s.fsm_x, estimate[0] - s.fsm_x)
            cmd_y = ctrl_y.compute_command(s.fsm_y, estimate[1] - s.fsm_y)
        if s.tick > warmup:
            recording.append((s.tick, s.theta_x, s.theta_y, s.fsm_x, s.fsm_y,
                              s.residual_x, s.residual_y, estimate[0], estimate[1]))
    data = np.array(recording)
    tracking_rmse = float(np.sqrt(np.mean(data[:, 5]**2 + data[:, 6]**2)))
    estimate_rmse = float(np.sqrt(np.mean(
        (data[:, 7] - data[:, 1])**2 + (data[:, 8] - data[:, 2])**2)))
    locked_percent = float(np.mean(np.hypot(data[:, 5], data[:, 6]) < 0.0002) * 100)
    return {'tracking_rmse_rad': tracking_rmse,
            'estimate_rmse_rad': estimate_rmse,
            'locked_percent': locked_percent,
            'samples': data}


def main():
    print('ROS-free closed-loop simulation, 200 Hz IMU / ~28.6 Hz camera / 20 ms delay')
    for mode in ('off', 'camera_only', 'naive', 'replay'):
        result = run_episode(mode)
        print('%-12s tracking RMSE=%7.4f mrad; estimate RMSE=%7.4f mrad; '
              'lock=%5.1f%%' % (
                  mode, 1000 * result['tracking_rmse_rad'],
                  1000 * result['estimate_rmse_rad'], result['locked_percent']))


if __name__ == '__main__':
    main()
