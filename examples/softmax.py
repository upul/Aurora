import numpy as np
import aurora.autodiff as ad
import aurora.datasets as data
from aurora.optim import SGD

D = 2
H = 150
K = 3
N = 100
X_data, y_data, y_data_encoded = data.spiral(K, D, N, 0)

x = ad.Variable(name="x")
y = ad.Variable(name='y')

W = ad.Parameter(name="W", init=np.zeros((D, K)))
b = ad.Parameter(name="b", init=np.zeros(K))

z = ad.matmul(x, W)
hid_1 = z + ad.broadcast_to(b, z)
loss = ad.cross_entropy(hid_1, y)

n_epoch = 1000
lr = 0.001

optimizer = SGD(loss, params=[W, b], lr=lr, momentum=0.9)

for i in range(n_epoch):
    loss_now = optimizer.step(feed_dict={x: X_data, y: y_data_encoded})
    if i % 100 == 0:
        fmt_str = 'iter: {0:>5d} cost: {1:>8.5f}'
        print(fmt_str.format(i, loss_now[0]))

prob = ad.softmax(hid_1)
executor = ad.Executor([prob])
prob_val, = executor.run(feed_dict={x: X_data})

correct = np.sum(np.equal(y_data, np.argmax(prob_val, axis=1)))
print('prediction accuracy: {}'.format(correct / (N * K)))
