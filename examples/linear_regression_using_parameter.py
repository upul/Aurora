import numpy as np
import aurora.autodiff as ad
import matplotlib.pyplot as plt
import seaborn as sbn;

sbn.set()
num_point = 250
n_epoch = 2500
lr = 0.01

X = ad.Variable(name='x')
y = ad.Variable(name='y')

W = ad.Parameter(name='W', init=np.zeros((1, 1)))
b = ad.Parameter(name='b', init=np.zeros(1))

z = ad.matmul(X, W)
output = z + ad.broadcast_to(b, z)

cost = ad.reduce_sum((y - output) * (y - output)) / (2.0 * num_point)
grad_cost_w, grad_b = ad.gradients(cost, [W, b])

x_data = np.linspace(0, 5, num_point).reshape((num_point, 1))
y_data = 2.0 * x_data + np.random.uniform(-1.0, 1.0, (num_point, 1)) + 2.5 * np.ones((num_point, 1))

executor = ad.Executor([cost, grad_cost_w, grad_b])

for i in range(n_epoch):
    # evaluate the graph
    cost_val, grad_w_val, grad_b_val = executor.run(feed_dict={X: x_data, y: y_data})
    if i % 100 == 0:
        print('iter: {0:>5d} cost: {1:>8.5f}'.format(i, cost_val[0]))
    W += -lr * grad_w_val
    b += -lr * grad_b_val

executor = ad.Executor([W, b])
W_val, b_val = executor.run(feed_dict={})
plt.scatter(x_data, y_data, c='#dd1c77')
plt.plot(x_data, x_data * W_val + b_val, c='#c994c7', linewidth=3)
plt.show()
