import aurora.autodiff as ad


class Base:
    def __init__(self, cost, params, lr=0.1):
        self.cost = cost
        self.params = params
        self.lr = lr
        grads = ad.gradients(cost, params)
        grads.insert(0, cost)
        self.executor = ad.Executor(grads)

    def step(self, feed_dict):
        raise NotImplementedError('This method should be implemented by subclasses')
