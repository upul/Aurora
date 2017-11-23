import numpy as np

from config import sys_configs
from .base import Base

if sys_configs['use_gpu']:
    from aurora.ndarray import ndarray, gpu_op


class Adam(Base):
    def __init__(self, cost, params, lr=1e-3, beta1=0.9, beta2=0.995, eps=1e-5, use_gpu=False):
        super().__init__(cost, params, lr, use_gpu=use_gpu)
        self.beta1 = beta1
        self.beta2 = beta2

        if self.use_gpu:
            self.velocity = [ndarray.array(np.zeros_like(param.const.asnumpy()), ctx=ndarray.gpu(0))
                             for param in params]
            self.momentum = [ndarray.array(np.zeros_like(param.const.asnumpy()), ctx=ndarray.gpu(0))
                             for param in params]

            self.vec_hat = [ndarray.array(np.zeros_like(param.const.asnumpy()), ctx=ndarray.gpu(0))
                            for param in self.params]
            self.mom_hat = [ndarray.array(np.zeros_like(param.const.asnumpy()), ctx=ndarray.gpu(0))
                            for param in self.params]
        else:
            self.velocity = [np.zeros_like(param.const) for param in params]
            self.momentum = [np.zeros_like(param.const) for param in params]

        self.time = 0
        self.eps = eps

    def step(self, feed_dict):
        exe_output = self.executor.run(feed_dict)
        self.time += 1

        if self.use_gpu:
            # set
            for i in range(len(self.vec_hat)):
                gpu_op.matrix_elementwise_multiply_by_const(self.vec_hat[i], 0.0, self.vec_hat[i])
                gpu_op.matrix_elementwise_multiply_by_const(self.mom_hat[i], 0.0, self.mom_hat[i])

            for i in range(len(self.params)):
                gpu_op.matrix_elementwise_multiply_by_const(self.momentum[i], self.beta1, self.momentum[i])

                # TODO: (upul) copying dev->hot>dev is expensive. We need a better approach.
                tem_gpu_array = ndarray.array(exe_output[i + 1].asnumpy(), ctx=ndarray.gpu(0))
                gpu_op.matrix_elementwise_multiply_by_const(exe_output[i + 1], (1 - self.beta1), tem_gpu_array)
                gpu_op.matrix_elementwise_add(self.momentum[i], tem_gpu_array, self.momentum[i])
                gpu_op.matrix_elementwise_div_by_const(self.momentum[i], (1 - self.beta1 ** self.time), self.mom_hat[i])

                gpu_op.matrix_elementwise_multiply_by_const(self.velocity[i], self.beta2, self.velocity[i])
                gpu_op.matrix_elementwise_multiply(exe_output[i + 1], exe_output[i + 1], exe_output[i + 1])
                gpu_op.matrix_elementwise_multiply_by_const(exe_output[i + 1], (1 - self.beta2), exe_output[i + 1])
                gpu_op.matrix_elementwise_add(self.velocity[i], exe_output[i + 1], self.velocity[i])
                gpu_op.matrix_elementwise_div_by_const(self.velocity[i], (1 - self.beta2 ** self.time), self.vec_hat[i])

            for i in range(len(self.params)):
                gpu_op.matrix_elementwise_sqrt(self.vec_hat[i], self.vec_hat[i])
                gpu_op.matrix_elementwise_add_by_const(self.vec_hat[i], self.eps, self.vec_hat[i])

                gpu_op.matrix_elementwise_multiply_by_const(self.mom_hat[i], -1 * self.lr, self.mom_hat[i])
                gpu_op.matrix_elementwise_division(self.mom_hat[i], self.vec_hat[i], self.mom_hat[i])
                gpu_op.matrix_elementwise_add(self.params[i].const, self.mom_hat[i], self.params[i].const)

        else:
            vec_hat = [np.zeros_like(param.const) for param in self.params]
            mom_hat = [np.zeros_like(param.const) for param in self.params]

            for i in range(len(self.params)):
                self.momentum[i] = self.beta1 * self.momentum[i] + (1 - self.beta1) * exe_output[i + 1]
                mom_hat[i] = self.momentum[i] / (1 - self.beta1 ** self.time)

                self.velocity[i] = self.beta2 * self.velocity[i] + (1 - self.beta2) * (exe_output[i + 1] ** 2)
                vec_hat[i] = self.velocity[i] / (1 - self.beta2 ** self.time)

            for i in range(len(self.params)):
                self.params[i].const += -self.lr * mom_hat[i] / (np.sqrt(vec_hat[i]) + self.eps)

        cost = exe_output[0]
        if self.use_gpu:
            cost = cost.asnumpy()
        return cost
