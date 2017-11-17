import aurora.autodiff as ad
from aurora.ndarray import ndarray

class Base:
    def __init__(self, cost, params, lr=0.1, use_gpu=False):
        self.cost = cost
        # if use_gpu == True, create matrices in GPU
        self.params = self._copy_to_gpu(params) if use_gpu else params
        self.lr = lr
        grads = ad.gradients(cost, params)
        grads.insert(0, cost)
        self.use_gpu = use_gpu
        self.executor = ad.Executor(grads, use_gpu=use_gpu)

    def step(self, feed_dict):
        raise NotImplementedError('This method should be implemented by subclasses')

    @staticmethod
    def _copy_to_gpu(params):
        ctx = ndarray.gpu(0)
        gpu_arrays = []
        for param in params:
            param.const = ndarray.array(param.const, ctx=ctx)
            gpu_arrays.append(param)
        return gpu_arrays

