import numpy as np
import aurora.autodiff as ad
from aurora.optim import SGD
import matplotlib.pyplot as plt
import seaborn as sbn

sbn.set()
num_point = 300
n_epoch = 1501
lr = 0.002

# X and y will be used to input into
# into the computational graph
X = ad.Variable(name='x')
y = ad.Variable(name='y')

# Parameters of the model
W = ad.Parameter(name='W', init=np.zeros((1, 1)))
b = ad.Parameter(name='b', init=np.zeros(1))

# Building linear regression model
z = ad.matmul(X, W)
output = z + ad.broadcast_to(b, z)
cost = ad.reduce_sum((y - output) * (y - output)) / (2.0 * num_point)

x_data = np.linspace(0, 5, num_point).reshape((num_point, 1))
y_data = 2.0 * x_data + np.random.uniform(-0.5, 0.5, (num_point, 1)) + 1.5 * np.ones((num_point, 1))

# Stochastic Gradient Descent is used to optimize
# parameters of the model
optimizer = SGD(cost, params=[W, b], lr=lr)
for i in range(n_epoch):
    cost_now = optimizer.step(feed_dict={X: x_data, y: y_data})
    if i <= 10 or (i <= 100 and i % 10 == 0) or (i <= 1000 and i % 100 == 0):
        fmt_str = 'iter: {0:>5d} cost: {1:>8.5f}'
        print(fmt_str.format(i, cost_now[0]))

# create an executor to read optimized W and b
executor = ad.Executor([W, b])
W_val, b_val = executor.run(feed_dict={})

# Plot training data points and learned parameters
plt.scatter(x_data, y_data, c='#dd1c77')
plt.plot(x_data, W_val * x_data + b_val, c='#c994c7', linewidth=4)
plt.show()
