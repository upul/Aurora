import aurora.autodiff as ad


class Base:
    def __init__(self, cost, optim_dict, lr=0.1):
        self.cost = cost
        self.optim_dict = optim_dict
        self.lr = lr
        grads = ad.gradients(cost, list(self.optim_dict.keys()))
        grads.insert(0, cost)
        self.executor = ad.Executor(grads)

    def step(self, feed_dict):
        raise NotImplementedError('This method should be implemented by subclasses')
