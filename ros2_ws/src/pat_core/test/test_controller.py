import math

def rms(values):
    return math.sqrt(
        sum(v * v for v in values) / len(values)
    )

def simulate_static_target(enabled):
    target = 0.001       # 1 mrad
    fsm_actual = 0.0
    fsm_command = 0.0

    dt = 0.001
    tau = 0.010         # 10 ms actuator time constant
    errors = []

    for k in range(2000):
        error = target - fsm_actual

        # Camera produces a measurement every ~33 ms.
        if enabled and k % 33 == 0:
            fsm_command = max(
                -0.002,
                min(0.002, fsm_actual + 0.7 * error)
            )

        # Approximate actuator dynamics.
        fsm_actual += (
            (fsm_command - fsm_actual) * dt / tau
        )

        errors.append(target - fsm_actual)

    return errors

def test_tracking_reduces_error():
    uncontrolled = simulate_static_target(False)
    controlled = simulate_static_target(True)

    uncontrolled_rms = rms(uncontrolled[-200:])
    controlled_rms = rms(controlled[-200:])

    assert controlled_rms < 0.1 * uncontrolled_rms
