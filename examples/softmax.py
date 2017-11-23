import numpy as np
import aurora as au
import aurora.autodiff as ad
import timeit
import argparse
import sys


def build_network(X, y, K):
    W = ad.Parameter(name="W", init=np.zeros((D, K)))
    b = ad.Parameter(name="b", init=np.zeros(K))

    z = ad.matmul(X, W)
    logit = z + ad.broadcast_to(b, z)
    loss = au.nn.cross_entropy_with_logits(logit, y)
    return loss, W, b, logit


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

    X = ad.Variable(name="x")
    y = ad.Variable(name='y')
    loss, W, b, logit = build_network(X, y, K)

    optimizer = au.optim.SGD(loss, params=[W, b], lr=1e-3, momentum=0.9, use_gpu=use_gpu)

    for i in range(n_iter):
        loss_now = optimizer.step(feed_dict={X: X_data, y: y_data_encoded})
        if i % 100 == 0:
            fmt_str = 'iter: {0:>5d} cost: {1:>8.5f}'
            print(fmt_str.format(i, loss_now[0]))

    prob = au.nn.softmax(logit)
    executor = ad.Executor([prob], use_gpu=use_gpu)
    prob_val, = executor.run(feed_shapes={X: X_data})

    if use_gpu:
        prob_val = prob_val.asnumpy()

    correct = np.sum(np.equal(y_data, np.argmax(prob_val, axis=1)))
    print('prediction accuracy: {0:.3f}'.format((correct / (N * K)) * 100))

    end = timeit.default_timer()
    print('\nTime taken for training/testing: {0:.3f}'.format(end - start))
