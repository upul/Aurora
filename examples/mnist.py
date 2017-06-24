import numpy as np
import aurora.autodiff as ad
from aurora.optim import Adam
from aurora.datasets import MNIST


def do_evaluation(preactivation, dataset):
    X_val, y_val = dataset
    prob = ad.softmax(preactivation)
    executor = ad.Executor([prob])
    prob_val, = executor.run(feed_dict={X: X_val})

    correct = np.sum(np.equal(y_val, np.argmax(prob_val, axis=1)))
    percentage = (correct / (y_val.shape[0])) * 100.00
    return percentage


dataset = MNIST(batch_size=128)
batch_generator = dataset.training_batch_generator()

input_size = dataset.num_features()
hid_1_size = 600
hid_2_size = 300
output_size = 10

X = ad.Variable(name="X")
y = ad.Variable(name='y')

W1 = ad.Parameter(name="W1", init=0.01 * np.random.randn(input_size, hid_1_size))
b1 = ad.Parameter(name="b1", init=np.zeros(hid_1_size))

W2 = ad.Parameter(name="W2", init=0.01 * np.random.randn(hid_1_size, hid_2_size))
b2 = ad.Parameter(name="b2", init=np.zeros(hid_2_size))

W3 = ad.Parameter(name="W3", init=0.01 * np.random.randn(hid_2_size, output_size))
b3 = ad.Parameter(name="b3", init=np.zeros(output_size))

z1 = ad.matmul(X, W1)
hidden_1 = z1 + ad.broadcast_to(b1, z1)
activation_1 = ad.relu(hidden_1)

z2 = ad.matmul(activation_1, W2)
hidden_2 = z2 + ad.broadcast_to(b2, z2)
activation_2 = ad.relu(hidden_2)

z3 = ad.matmul(activation_2, W3)
hidden_3 = z3 + ad.broadcast_to(b3, z3)
loss = ad.cross_entropy(hidden_3, y)

lr = 1e-3
n_epoch = 10001
optimizer = Adam(loss, params=[W1, b1, W2, b2, W3, b3], lr=lr)
for i in range(n_epoch):
    X_batch, y_batch = next(batch_generator)
    loss_now = optimizer.step(feed_dict={X: X_batch, y: y_batch})
    if i <= 10 or (i <= 100 and i % 10 == 0) or (i <= 1000 and i % 100 == 0) or (i <= 10000 and i % 500 == 0):
        fmt_str = 'iter: {0:>5d} cost: {1:>8.5f}'
        print(fmt_str.format(i, loss_now[0]))

val_acc = do_evaluation(hidden_3, dataset.validation())
print('Validation accuracy: {:>.2f}'.format(val_acc))

test_acc = do_evaluation(hidden_3, dataset.testing())
print('Testing accuracy: {:>.2f}'.format(test_acc))
