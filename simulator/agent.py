import numpy as np


class Agent:

    def __init__(self, position, velocity, radius):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.radius = radius

    def update(self, dt):
        self.position += self.velocity * dt
