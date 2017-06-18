import numpy as np
from .base import Base


class SGD(Base):
    def __init__(self, cost, optim_dict, lr=0.1, momentum=0.9):
        super().__init__(cost, optim_dict, lr)
        self.momentum = momentum
        self.velocity = {key: np.zeros_like(optim_dict[key]) for key in optim_dict.keys()}

    def step(self, feed_dict):
        feed_data = feed_dict.copy()
        for param in list(self.optim_dict.keys()):
            feed_data[param] = self.optim_dict[param]
        exe_output = self.executor.run(feed_data)

        parameters = list(self.optim_dict.keys())
        for index in range(len(parameters)):
            self.velocity[parameters[index]] = \
                self.momentum * self.velocity[parameters[index]] - self.lr * exe_output[index + 1]
            self.optim_dict[parameters[index]] += self.velocity[parameters[index]]

        step_param = self.optim_dict.copy()
        step_param[self.cost] = exe_output[0]
        return step_param
