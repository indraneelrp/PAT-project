import math
from pat_core.controller import FSMController
from pat_core.simulator_core import PATSimulator
from pat_core.offline_validation import run_episode


def test_camera_observes_actuator_motion():
    # No platform motion assumption necessary: reconstruct theta from the
    # captured residual and actual mirror position.
    sim = PATSimulator(seed=9, latency_ms=0)
    for _ in range(sim.camera_ticks):
        s = sim.step(command_x=0.001, command_y=-0.0004)
    assert s.camera
    c = s.camera[0]
    assert abs((c.error_x + c.fsm_x_at_capture) - s.theta_x) < 0.0002
    assert abs((c.error_y + c.fsm_y_at_capture) - s.theta_y) < 0.0002


def test_controller_clamps():
    ctrl = FSMController(gain=0.7, limit=0.002)
    assert ctrl.compute_command(0.0, 0.1) == 0.002
    assert ctrl.compute_command(0.0, -0.1) == -0.002


def test_replay_tracking_beats_open_loop():
    off = run_episode('off')
    replay = run_episode('replay')
    assert replay['tracking_rmse_rad'] < off['tracking_rmse_rad'] * 0.50


def test_same_seed_is_deterministic():
    a = run_episode('replay', steps=800)
    b = run_episode('replay', steps=800)
    assert math.isclose(a['tracking_rmse_rad'], b['tracking_rmse_rad'])
