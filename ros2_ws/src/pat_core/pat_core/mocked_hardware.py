'''
Provide the simulated physical response of components
'''

class FirstOrderFSM:
    def __init__(self, tau=0.010):
        self.tau = tau
        self.position = 0.0

    def step(self, command, dt):
        self.position += (
            (command - self.position) * dt / self.tau
        )

        return self.position