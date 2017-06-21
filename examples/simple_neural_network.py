import numpy as np
import aurora.autodiff as ad
from aurora.optim import SGD

np.random.seed(0)
N = 100  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
H = 150
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

W1 = ad.Parameter(name="W1", init=0.01 * np.random.randn(D, H))
b1 = ad.Parameter(name="b1", init=np.zeros(H))

W2 = ad.Parameter(name="W2", init=0.01 * np.random.randn(H, K))
b2 = ad.Parameter(name="b2", init=np.zeros(K))

z1 = ad.matmul(X, W1)
hidden_1 = z1 + ad.broadcast_to(b1, z1)
activation_1 = ad.relu(hidden_1)
z2 = ad.matmul(activation_1, W2)
hidden_2 = z2 + ad.broadcast_to(b2, z2)
loss = ad.cross_entropy(hidden_2, y)

lr = 2e-3
n_epoch = 10001
optimizer = SGD(loss, params=[W1, b1, W2, b2], lr=lr, momentum=0.9)
for i in range(n_epoch):
    loss_now = optimizer.step(feed_dict={X: X_data, y: y_one_hot})
    if i <= 10 or (i <= 100 and i % 10 == 0) or (i <= 1000 and i % 100 == 0):
        fmt_str = 'iter: {0:>5d} cost: {1:>8.5f}'
        print(fmt_str.format(i, loss_now[0]))

prob = ad.softmax(hidden_2)
executor = ad.Executor([prob])
prob_values, = executor.run(feed_dict={X: X_data})

correct = np.sum(np.equal(y_data, np.argmax(prob_values, axis=1)))
print('Prediction accuracy: {0:>.3f}%'.format((correct / (N * K)) * 100.00))
