import numpy as np
import aurora.autodiff as ad
from aurora.optim import Adam
from aurora.optim import SGD
import matplotlib.pyplot as plt
import seaborn as sbn;

np.random.seed(0)
N = 100  # number of points per class
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

X = ad.Variable(name="X")
y = ad.Variable(name='y')

W1 = ad.Variable(name="W1")
b1 = ad.Variable(name="b1")

W2 = ad.Variable(name="W2")
b2 = ad.Variable(name="b2")

z1 = ad.matmul(X, W1)
hid1 = z1 + ad.broadcast_to(b1, z1)
act1 = ad.relu(hid1)

z2 = ad.matmul(act1, W2)
hid2 = z2 + ad.broadcast_to(b2, z2)

loss = ad.cross_entropy(hid2, y)

h = 150  # size of hidden layer
W1_val = 0.01 * np.random.randn(D, h)
b1_val = np.zeros(h)
W2_val = 0.01 * np.random.randn(h, K)
b2_val = np.zeros(K)

learning_rate = 1e-3
n_epoch = 1000
optimizer = SGD(loss, optim_dict={W1: W1_val,
                                   b1: b1_val,
                                   W2: W2_val,
                                   b2: b2_val}, lr=learning_rate)
for i in range(n_epoch):
    step_params = optimizer.step(feed_dict={X: X_data, y: y_one_hot})
    if i % 50 == 0:
        fmt_str = 'iter: {0:>5d} cost: {1:>8.5f}'
        print(fmt_str.format(i, step_params[loss][0]))

softmax = ad.softmax(hid2)
executor = ad.Executor([softmax])
a, = executor.run(feed_dict={X: X_data,
                             W1: step_params[W1],
                             b1: step_params[b1],
                             W2: step_params[W2], b2: step_params[b2]})

correct = np.sum(np.equal(y_data, np.argmax(a, axis=1)))
print('Prediction accuracy: {0:>.3f}%'.format((correct / (N * K)) * 100.00))
