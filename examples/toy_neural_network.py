import numpy as np
import aurora as au
import aurora.autodiff as ad
import seaborn as sbn
sbn.set()

D = 2
H = 150
K = 3
N = 100
X_data, y_data, y_data_encoded = au.datasets.spiral(K, D, N, 0)

X = ad.Variable(name="X")
y = ad.Variable(name='y')

W1 = ad.Parameter(name="W1", init=0.01 * np.random.randn(D, H))
b1 = ad.Parameter(name="b1", init=np.zeros(H))

W2 = ad.Parameter(name="W2", init=0.01 * np.random.randn(H, K))
b2 = ad.Parameter(name="b2", init=np.zeros(K))

z1 = ad.matmul(X, W1)
hidden_1 = z1 + ad.broadcast_to(b1, z1)
activation_1 = au.nn.relu(hidden_1)

z2 = ad.matmul(activation_1, W2)
hidden_2 = z2 + ad.broadcast_to(b2, z2)
loss = au.nn.cross_entropy_with_logits(hidden_2, y)

lr = 1e-3
n_epoch = 1001
optimizer = au.optim.SGD(loss, params=[W1, b1, W2, b2], lr=lr, momentum=0.8)
for i in range(n_epoch):
    loss_now = optimizer.step(feed_dict={X: X_data, y: y_data_encoded})
    if i <= 10 or (i <= 100 and i % 10 == 0) or (i <= 1000 and i % 100 == 0):
        fmt_str = 'iter: {0:>5d} cost: {1:>8.5f}'
        print(fmt_str.format(i, loss_now[0]))

prob = au.nn.softmax(hidden_2)
executor = ad.Executor([prob])
prob_values, = executor.run(feed_dict={X: X_data})

correct = np.sum(np.equal(y_data, np.argmax(prob_values, axis=1)))
print('Prediction accuracy: {0:>.3f}%'.format((correct / (N * K)) * 100.00))
