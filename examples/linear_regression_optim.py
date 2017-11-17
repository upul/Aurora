import numpy as np
import aurora.autodiff as ad
from aurora.optim import SGD
import matplotlib.pyplot as plt
import seaborn as sbn
import argparse
import timeit

sbn.set()


def build_graph(X, y):
    # Parameters of the model
    W = ad.Parameter(name='W', init=np.zeros((1, 1)))
    b = ad.Parameter(name='b', init=np.zeros(1))

    # Building linear regression model
    z = ad.matmul(X, W)
    output = z + ad.broadcast_to(b, z)
    loss = ad.reduce_sum((y - output) * (y - output)) / (2.0 * num_point)

    return loss, W, b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--exe_context',
                        help='Choose execution context: numpy, gpu',
                        default='numpy')

    parser.add_argument('-i', '--num_iter',
                        help='Choose number of iterations',
                        default=500)

    args = parser.parse_args()

    num_point = 500
    lr = 0.002

    use_gpu = False
    if args.exe_context == 'gpu':
        use_gpu = True
    n_iter = int(args.num_iter)

    start = timeit.default_timer()

    # X and y will be used to input into
    # into the computational graph
    X = ad.Variable(name='x')
    y = ad.Variable(name='y')

    loss, W, b = build_graph(X, y)

    x_data = np.linspace(0, 5, num_point).reshape((num_point, 1))
    y_data = 2.0 * x_data + np.random.uniform(-0.5, 0.5, (num_point, 1)) + 1.5 * np.ones((num_point, 1))

    # Stochastic Gradient Descent is used to optimize
    # parameters of the model
    optimizer = SGD(loss, params=[W, b], lr=lr, use_gpu=True)

    for i in range(n_iter):
        cost_now = optimizer.step(feed_dict={X: x_data, y: y_data})
        if i <= 10 or (i <= 100 and i % 10 == 0) or (i <= 1000 and i % 100 == 0):
            fmt_str = 'iter: {0:>5d} cost: {1:>8.5f}'
            print(fmt_str.format(i, cost_now[0]))

    # create an executor to read optimized W and b
    executor = ad.Executor([W, b], use_gpu=True)
    W_val, b_val = executor.run(feed_shapes={})

    W_val = W_val.asnumpy()
    b_val = b_val.asnumpy()
    end = timeit.default_timer()
    print('\nTime taken for training/testing: {0:3f}'.format(end - start))

    # Plot training data points and learned parameters
    plt.scatter(x_data, y_data, c='#dd1c77')
    plt.plot(x_data, W_val * x_data + b_val, c='#c994c7', linewidth=4)
    plt.show()
