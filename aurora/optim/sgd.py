import numpy as np
from .base import Base


class SGD(Base):
    def __init__(self, cost, params, lr=0.1, momentum=0.9):
        super().__init__(cost, params, lr)
        self.momentum = momentum
        self.velocity = self._init_velocity_vec(params)

    def step(self, feed_dict):
        exe_output = self.executor.run(feed_dict)
        for i in range(len(self.params)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * exe_output[1 + i]
            self.params[i].const += self.velocity[i]
        return exe_output[0]

    @staticmethod
    def _init_velocity_vec(params):
        vector = []
        for param in params:
            vector.append(np.zeros_like(param.const))
        return vector
