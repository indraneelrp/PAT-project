import math
from pat_core.controller import FSMController
from ros2_ws.src.pat_core.pat_core.mocked_hardware import FirstOrderFSM


def simulate_static_target(enabled):
    target = 0.001

    controller = FSMController(gain=0.7, limit=0.002)
    fsm = FirstOrderFSM(tau=0.010)

    fsm_command = 0.0
    dt = 0.001
    errors = []

    for k in range(2000):
        error = target - fsm.position

        if enabled and k % 33 == 0:
            fsm_command = controller.compute_command(
                fsm.position, error
            )

        fsm.step(fsm_command, dt)

        errors.append(target - fsm.position)

    return errors


def rms(values):
    return math.sqrt(
        sum(v * v for v in values) / len(values)
    )


def test_tracking_reduces_error():
    uncontrolled = simulate_static_target(False)
    controlled = simulate_static_target(True)

    uncontrolled_rms = rms(uncontrolled[-200:])
    controlled_rms = rms(controlled[-200:])

    assert controlled_rms < 0.1 * uncontrolled_rms
