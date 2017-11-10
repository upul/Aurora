import numpy as np
import aurora.autodiff as ad
import matplotlib.pyplot as plt
import seaborn as sbn

sbn.set()
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
executor = ad.Executor([cost, grad_cost_w, grad_b])

for i in range(n_epoch):
    # evaluate the graph
    cost_val, grad_cost_w_val, grad_b_val = executor.run(feed_shapes={x: x_data, W: w_val, y: y_data, b: b_val})
    if i <= 10 or (i <= 100 and i % 10 == 0) or (i <= 1000 and i % 100 == 0):
        fmt_str = 'iter: {0:>5d} cost: {1:>8.5f}, W: {2:>8.5f} b: {3:>8.5f}'
        print(fmt_str.format(i, cost_val[0], w_val[0, 0], b_val[0]))

    w_val += -lr * grad_cost_w_val
    b_val += -lr * grad_b_val

plt.scatter(x_data, y_data, c='#dd1c77')
plt.plot(x_data, w_val[0, 0] * x_data + b_val[0], c='#c994c7', linewidth=3)
plt.show()
