import numpy as np
import aurora.autodiff as ad
import matplotlib.pyplot as plt
import seaborn as sbn
from aurora.ndarray import ndarray, gpu_op

sbn.set()


def sgd_update_gpu(param, grad_param, learning_rate):
    assert isinstance(param, ndarray.NDArray)
    assert isinstance(grad_param, ndarray.NDArray)

    gpu_op.matrix_elementwise_multiply_by_const(grad_param, -learning_rate, grad_param)
    gpu_op.matrix_elementwise_add(param, grad_param, param)


num_point = 250
n_epoch = 2500
lr = 0.04

x = ad.Variable(name='x')
y = ad.Variable(name='y')
W = ad.Variable(name='W')
b = ad.Variable(name='b')
z = ad.matmul(x, W)
output = z + ad.broadcast_to(b, z)

cost = ad.reduce_sum((y - output) * (y - output)) / (2.0 * num_point)
grad_cost_w, grad_b = ad.gradients(cost, [W, b])

x_data = np.linspace(0, 5, num_point).reshape((num_point, 1))
y_data = 2.0 * x_data + np.random.uniform(-1.0, 1.0, (num_point, 1)) + 2.5 * np.ones((num_point, 1))

w_val = np.zeros((1, 1))
b_val = np.zeros(1)
executor_ctx = ndarray.gpu(0)

w_val = ndarray.array(w_val, ctx=executor_ctx)
b_val = ndarray.array(b_val, ctx=executor_ctx)

x_data = ndarray.array(x_data, ctx=executor_ctx)
y_data = ndarray.array(y_data, ctx=executor_ctx)

executor = ad.Executor([cost, grad_cost_w, grad_b], use_gpu=True)

for i in range(n_epoch):
    # evaluate the graph
    cost_val, grad_w_val, grad_b_val = executor.run(feed_shapes={x: x_data, W: w_val, y: y_data, b: b_val})

    cost_val_np = cost_val.asnumpy()
    w_val_np = w_val.asnumpy()
    b_val_np = b_val.asnumpy()
    if i <= 10 or (i <= 100 and i % 10 == 0) or (i <= 1000 and i % 100 == 0):
        fmt_str = 'iter: {0:>5d} cost: {1:>8.5f}, W: {2:>8.5f} b: {3:>8.5f}'
        print(fmt_str.format(i, cost_val_np[0], w_val_np[0, 0], b_val_np[0]))

    sgd_update_gpu(w_val, grad_w_val, lr)
    sgd_update_gpu(b_val, grad_b_val, lr)

x_data_np = x_data.asnumpy()
y_data_np = y_data.asnumpy()
plt.scatter(x_data_np, y_data_np, c='#dd1c77')

w_val_np = w_val.asnumpy()
b_val_np = b_val.asnumpy()
plt.plot(x_data_np, w_val_np[0, 0] * x_data_np + b_val_np[0], c='#c994c7', linewidth=3)
plt.show()
