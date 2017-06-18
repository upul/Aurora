import numpy as np
import aurora.autodiff as ad
from aurora.optim import Adam
import matplotlib.pyplot as plt
import seaborn as sbn;

sbn.set()
num_point = 300
n_epoch = 1500
lr = 0.06

x = ad.Variable(name='x')
y = ad.Variable(name='y')
W = ad.Variable(name='W')
b = ad.Variable(name='b')
z = ad.matmul(x, W)
output = z + ad.broadcast_to(b, z)

cost = ad.reduce_sum((y - output) * (y - output)) / (2.0 * num_point)
grad_cost_w, grad_b = ad.gradients(cost, [W, b])

x_data = np.linspace(0, 5, num_point).reshape((num_point, 1))
y_data = 2.0 * x_data + np.random.uniform(-0.5, 0.5, (num_point, 1)) + 1.5 * np.ones((num_point, 1))

w_val = np.zeros((1, 1))
b_val = np.zeros(1)
executor = ad.Executor([cost, grad_cost_w, grad_b])

optimizer = Adam(cost, optim_dict={W: w_val, b: b_val}, lr=lr)
for i in range(n_epoch):
    step_params = optimizer.step(feed_dict={x: x_data, y: y_data})
    if i % 100 == 0:
        print('iter: {0:>5d} cost: {1:>8.5f}, W: {2:>8.5f} b: {3:>8.5f}'.format(i,
                                                                                step_params[cost][0],
                                                                                step_params[W][0, 0],
                                                                                step_params[b][0]))

plt.scatter(x_data, y_data, c='#dd1c77')
plt.plot(x_data, step_params[W][0, 0] * x_data + step_params[b][0], c='#c994c7', linewidth=4)
plt.show()
