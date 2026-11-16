"""Two-state fine tracking supervisor, independent of ROS.

FINE_LOCK: actuator is trying to achieve sufficiently small residual error.
TRACK:     actuator has maintained small residual error long enough for lock.

This is not a complete PAT SEARCH/ACQUIRE state machine.  A camera time-out
forces TRACK -> FINE_LOCK; it does not command a coarse gimbal search.
"""
import math


class FineTrackingStateMachine:
    DISABLED = 'DISABLED'
    FINE_LOCK = 'FINE_LOCK'
    TRACK = 'TRACK'

    def __init__(self, lock_threshold=0.0002, loss_threshold=0.0004,
                 lock_samples=15, loss_samples=5):
        if not (0 < lock_threshold < loss_threshold):
            raise ValueError('Expected 0 < lock_threshold < loss_threshold')
        if lock_samples < 1 or loss_samples < 1:
            raise ValueError('Sample counts must be positive')
        self.lock_threshold = float(lock_threshold)
        self.loss_threshold = float(loss_threshold)
        self.lock_samples = int(lock_samples)
        self.loss_samples = int(loss_samples)
        self.state = self.DISABLED
        self.good_count = 0
        self.bad_count = 0

    def set_enabled(self, enabled):
        """Enable starts in FINE_LOCK; disable returns to DISABLED."""
        new_state = self.FINE_LOCK if enabled else self.DISABLED
        changed = new_state != self.state
        self.state = new_state
        self.good_count = 0
        self.bad_count = 0
        return changed

    def update(self, residual_x, residual_y, camera_fresh=True):
        """Evaluate *estimated* pointing residual; never use simulator truth.

        Return True if the state changed on this update. Only transition
        after consecutive samples; this avoids mode chatter due to noise.
        """
        if self.state == self.DISABLED:
            return False
        if not camera_fresh:
            self.good_count = 0
            self.bad_count = 0
            if self.state == self.TRACK:
                self.state = self.FINE_LOCK
                return True
            return False

        error = math.hypot(float(residual_x), float(residual_y))
        if not math.isfinite(error):
            # Treat invalid estimates as lost lock.
            return self.update(0.0, 0.0, camera_fresh=False)

        if self.state == self.FINE_LOCK:
            self.good_count = self.good_count + 1 if error < self.lock_threshold else 0
            if self.good_count >= self.lock_samples:
                self.state = self.TRACK
                self.good_count = 0
                self.bad_count = 0
                return True
        else:  # TRACK
            self.bad_count = self.bad_count + 1 if error > self.loss_threshold else 0
            if self.bad_count >= self.loss_samples:
                self.state = self.FINE_LOCK
                self.good_count = 0
                self.bad_count = 0
                return True
        return False
