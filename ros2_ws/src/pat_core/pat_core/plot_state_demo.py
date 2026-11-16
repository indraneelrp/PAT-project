"""Plot the standalone PAT FINE_LOCK/TRACK demonstration (no ROS required).

Run from your ROS workspace after sourcing Python package path:
  export PYTHONPATH="$PWD/src/pat_core:$PYTHONPATH"
  python3 plot_state_demo.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pat_core.simulator_core import PATSimulator
from pat_core.estimator import DelayAwareKalmanEstimator2D
from pat_core.controller import FSMController
from pat_core.state_machine import FineTrackingStateMachine


def main():
    plant = PATSimulator(seed=42)
    est = DelayAwareKalmanEstimator2D(history_capacity=500)
    for axis in (est.kf.x_axis, est.kf.y_axis):
        axis.R = plant.camera_std ** 2
        axis.P = np.diag([1e-6, 1e-6])
        axis.Q = np.diag([1e-11, 1e-9])
    ctrl_x, ctrl_y = FSMController(), FSMController()
    sm = FineTrackingStateMachine()
    sm.set_enabled(True)
    cmd_x = cmd_y = 0.0
    last_camera_tick = None
    last_camera_error = None
    t, true_errors, estimated_errors, modes = [], [], [], []
    events = [(0, sm.state)]

    for _ in range(1600):
        if plant.tick == 600:
            plant.offset_enabled = True
        elif plant.tick == 950:
            plant.offset_enabled = False

        sample = plant.step(cmd_x, cmd_y)
        est.predict(sample.dt, sample.gyro_x, sample.gyro_y)
        for cam in sample.camera:
            if cam.detected:
                last_camera_tick = cam.capture_tick
                last_camera_error = float(np.hypot(cam.error_x, cam.error_y))
                est.update_camera_at(
                    cam.capture_tick,
                    cam.error_x + cam.fsm_x_at_capture,
                    cam.error_y + cam.fsm_y_at_capture,
                )
        theta_x, theta_y = est.angle
        ex = theta_x - sample.fsm_x
        ey = theta_y - sample.fsm_y
        estimated_error = float(np.hypot(ex, ey))
        camera_fresh = (
            last_camera_tick is not None
            and (sample.tick - last_camera_tick) * sample.dt <= 0.15
        )
        mode_error = max(estimated_error, last_camera_error or 0.0)
        if sm.update(mode_error, 0.0, camera_fresh):
            events.append((sample.tick, sm.state))

        cmd_x = ctrl_x.compute_command(sample.fsm_x, ex)
        cmd_y = ctrl_y.compute_command(sample.fsm_y, ey)
        t.append(sample.tick * sample.dt)
        true_errors.append(1000.0 * np.hypot(sample.residual_x, sample.residual_y))
        estimated_errors.append(1000.0 * estimated_error)
        modes.append(sm.state)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, true_errors, label='Actual pointing error (ground truth)', linewidth=1.3)
    ax.plot(t, estimated_errors, label='Estimated residual error', linewidth=1, alpha=0.65)
    ax.axhline(0.2, linestyle='--', color='gray', label='Lock threshold (0.2 mrad)')
    ax.axhline(0.4, linestyle=':', color='gray', label='Loss threshold (0.4 mrad)')
    ax.axvspan(600 * plant.dt, 950 * plant.dt, alpha=0.12, color='orange', label='Injected target offset')
    for tick, mode in events[1:]:
        ax.axvline(tick * plant.dt, linewidth=0.65, alpha=0.4, color='purple')
    ax.set(xlabel='Simulation time (s)', ylabel='2D pointing error (mrad)',
           title='Fine-pointing lock, disturbance and recovery')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    filename = 'pat_state_demo_error.png'
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    print(f'Saved {filename}')
    for tick, mode in events:
        print(f'tick {tick:4d} | {tick * plant.dt:5.2f} s | {mode}')


if __name__ == '__main__':
    main()
