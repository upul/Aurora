import numpy as np
import aurora.autodiff as ad
from aurora.optim import Adam, SGD
import matplotlib.pyplot as plt
import seaborn as sbn;

np.random.seed(0)
N = 250  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
X_data = np.zeros((N * K, D))
y_data = np.zeros(N * K, dtype='uint8')
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X_data[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y_data[ix] = j

y_one_hot = np.zeros((N * K, K))
y_one_hot[range(N * K), y_data] = 1

######################################
x = ad.Variable(name="x")
y = ad.Variable(name='y')
W = ad.Variable(name="w")
b = ad.Variable(name="b")

z = ad.matmul(x, W)
softmax = z + ad.broadcast_to(b, z)

loss = ad.cross_entropy(softmax, y)
w_val = np.zeros((D, K))
b_val = np.zeros(K)

n_epoch = 1500
lr = 0.01

optimizer = Adam(loss, optim_dict={W: w_val, b: b_val}, lr=lr)

#grad_w, grad_b = ad.gradients(loss, [W, b])
#executor = ad.Executor([loss, grad_w, grad_b])

for i in range(n_epoch):
    step_params = optimizer.step(feed_dict={x: X_data, y: y_one_hot})
    if i % 100 == 0:
        fmt_str = 'iter: {0:>5d} cost: {1:>8.5f}, W: {2:>8.5f} b: {3:>8.5f}'
        print(fmt_str.format(i,
                             step_params[loss][0],
                             step_params[W][0, 0],
                             step_params[b][0]))
softmax = ad.softmax(softmax)
executor = ad.Executor([softmax])
a, = executor.run(feed_dict={x: X_data, W: step_params[W], b: step_params[b]})

correct = np.sum(np.equal(y_data, np.argmax(a, axis=1)))
print('prediction accuracy: {}'.format(correct / (N * K)))
