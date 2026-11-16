"""Integration of plant, delayed Kalman estimator, controller and state machine."""
from pat_core.state_demo import run_demo


def test_transitions_are_demonstrable():
    events = run_demo()
    # The initial residual converges and the supervisor declares TRACK.
    assert any(0 < tick < 600 and mode == 'TRACK' for tick, mode in events), events
    # A step in target angle is injected at tick 600; lock can be lost.
    assert any(600 < tick < 950 and mode == 'FINE_LOCK' for tick, mode in events), events
    # Step removed at tick 950, eventually stabilising within fine range.
    assert any(tick > 950 and mode == 'TRACK' for tick, mode in events), events
