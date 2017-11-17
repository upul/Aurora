import numpy as np
import aurora as au
import aurora.autodiff as ad
import seaborn as sbn
import timeit
import argparse

sbn.set()


def build_network(X, y, H, K):
    W1 = ad.Parameter(name="W1", init=0.01 * np.random.randn(D, H))
    b1 = ad.Parameter(name="b1", init=np.zeros(H))

    W2 = ad.Parameter(name="W2", init=0.01 * np.random.randn(H, K))
    b2 = ad.Parameter(name="b2", init=np.zeros(K))

    z1 = ad.matmul(X, W1)
    hidden_1 = z1 + ad.broadcast_to(b1, z1)
    activation_1 = au.nn.relu(hidden_1)

    z2 = ad.matmul(activation_1, W2)
    logit = z2 + ad.broadcast_to(b2, z2)
    loss = au.nn.cross_entropy_with_logits(logit, y)

    return loss, W1, b1, W2, b2, logit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--exe_context',
                        help='Choose execution context: numpy, gpu',
                        default='numpy')

    parser.add_argument('-i', '--num_iter',
                        help='Choose number of iterations',
                        default=500)

    args = parser.parse_args()

    use_gpu = False
    if args.exe_context == 'gpu':
        use_gpu = True
    n_iter = int(args.num_iter)

    start = timeit.default_timer()
    D = 2
    H = 150
    K = 3
    N = 100
    X_data, y_data, y_data_encoded = au.datasets.spiral(K, D, N, 0)

    X = ad.Variable(name='X')
    y = ad.Variable(name='y')
    loss, W1, b1, W2, b2, logit = build_network(X, y, H, K)

    optimizer = au.optim.SGD(loss, params=[W1, b1, W2, b2], lr=1e-3, momentum=0.8, use_gpu=use_gpu)
    for i in range(n_iter):
        loss_now = optimizer.step(feed_dict={X: X_data, y: y_data_encoded})
        if i <= 10 or (i <= 100 and i % 10 == 0) or (i <= 1000 and i % 100 == 0):
            fmt_str = 'iter: {0:>5d} cost: {1:>8.5f}'
            print(fmt_str.format(i, loss_now[0]))

    prob = au.nn.softmax(logit)
    executor = ad.Executor([prob], use_gpu=use_gpu)
    prob_values, = executor.run(feed_shapes={X: X_data})
    if use_gpu:
        prob_values = prob_values.asnumpy()

    correct = np.sum(np.equal(y_data, np.argmax(prob_values, axis=1)))
    print('Prediction accuracy: {0:>.3f}%'.format((correct / (N * K)) * 100.00))

    end = timeit.default_timer()
    print('\nTime taken for training/testing: {0:3f}'.format(end - start))
