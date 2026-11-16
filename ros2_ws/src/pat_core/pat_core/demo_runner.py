"""Deterministic no-ROS demonstration of both supervisory state transitions.

python3 -m pat_core.demo_runner

ROS integration uses the exact same state_machine.py class in controller_node.
"""
import numpy as np
from .simulator_core import PATSimulator
from .estimator import DelayAwareKalmanEstimator2D
from .controller import FSMController
from .state_machine import FineTrackingStateMachine


def run_demo():
    plant = PATSimulator(seed=42)
    est = DelayAwareKalmanEstimator2D(history_capacity=500)
    for axis in (est.kf.x_axis, est.kf.y_axis):
        axis.R = plant.camera_std ** 2
        axis.P = np.diag([1e-6, 1e-6])
        axis.Q = np.diag([1e-11, 1e-9])
    ctrl_x, ctrl_y = FSMController(), FSMController()
    sm = FineTrackingStateMachine()
    sm.set_enabled(True)
    cmd_x, cmd_y = 0.0, 0.0
    last_camera_tick = None
    last_camera_error = None
    events = [(0, sm.state)]

    for _ in range(1600):
        # After initial lock, step beyond FSM range then restore alignment.
        if plant.tick == 600:
            plant.offset_enabled = True
        elif plant.tick == 950:
            plant.offset_enabled = False
        s = plant.step(cmd_x, cmd_y)
        est.predict(s.dt, s.gyro_x, s.gyro_y)
        for c in s.camera:
            if c.detected:
                last_camera_tick = c.capture_tick
                last_camera_error = float(np.hypot(c.error_x, c.error_y))
                est.update_camera_at(
                    c.capture_tick,
                    c.error_x + c.fsm_x_at_capture,
                    c.error_y + c.fsm_y_at_capture)
        theta_x, theta_y = est.angle
        ex, ey = theta_x - s.fsm_x, theta_y - s.fsm_y
        camera_fresh = (last_camera_tick is not None and
                        (s.tick - last_camera_tick) * s.dt <= 0.15)
        mode_error = max(float(np.hypot(ex, ey)), last_camera_error or 0.0)
        if sm.update(mode_error, 0.0, camera_fresh):
            events.append((s.tick, sm.state))
        cmd_x = ctrl_x.compute_command(s.fsm_x, ex)
        cmd_y = ctrl_y.compute_command(s.fsm_y, ey)

    return events


def main():
    for tick, state in run_demo():
        print(f'tick {tick:4d} | {tick / 200:5.2f} s | {state}')


if __name__ == '__main__':
    main()
