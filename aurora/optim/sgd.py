import aurora.autodiff as ad
from .base import Base


class SGD(Base):
    def __init__(self, cost, optim_dict, lr=0.1, momentum=0.9):
        super().__init__(cost, optim_dict, lr, momentum)

    def step(self, feed_dict):
        feed_data = feed_dict.copy()
        for param in list(self.optim_dict.keys()):
            feed_data[param] = self.optim_dict[param]
        exe_output = self.executor.run(feed_data)

        parameters = list(self.optim_dict.keys())
        for index in range(len(parameters)):
            self.optim_dict[parameters[index]] += -self.lr * exe_output[index + 1]

        step_param = self.optim_dict.copy()
        step_param[self.cost] = exe_output[0]
        return step_param
