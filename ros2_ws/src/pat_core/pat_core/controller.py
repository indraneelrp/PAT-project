class FSMController:
    def __init__(self, gain=0.7, limit=0.002):
        self.gain = gain
        self.limit = limit

    def compute_command(self, current_position, error):
        desired = current_position + self.gain * error

        return max(
            -self.limit,
            min(self.limit, desired)
        )