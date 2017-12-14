"""Trains a simple convnet on the MNIST dataset.
=====================================================================
Numpy:   Gets 98.00 % test accuracy after 4500 iterations with
         64 batch size.

         Running Time: 1149.08 seconds on Intel(R) Core(TM) i7-7700K
         CPU @ 4.20GHz 8 Cores.


GPU:    Will be added later
"""

import argparse
import timeit

import aurora as au
import aurora.autodiff as ad
import numpy as np


def build_network(image, y, batch_size=32):
    rand = np.random.RandomState(seed=1024)

    reshaped_images = ad.reshape(image, newshape=(batch_size, 1, 28, 28))

    # weight in (number_kernels, color_depth, kernel_height, kernel_width)
    W1 = ad.Parameter(name='W1', init=rand.normal(scale=0.1, size=(10, 1, 5, 5)))
    b1 = ad.Parameter(name='b1', init=rand.normal(scale=0.1, size=10))
    conv1 = au.nn.conv2d(input=reshaped_images, filter=W1, bias=b1)
    activation1 = au.nn.relu(conv1)
    # size of activation1: batch_size x 10 x 24 x 24

    # weight in (number_kernels, number_kernels of previous layer, kernel_height, kernel_width)
    W2 = ad.Parameter(name='W2', init=rand.normal(scale=0.1, size=(5, 10, 5, 5)))
    b2 = ad.Parameter(name='b2', init=rand.normal(scale=0.1, size=5))
    conv2 = au.nn.conv2d(input=activation1, filter=W2, bias=b2)
    activation2 = au.nn.relu(conv2)
    # size of activation2: batch_size x 5 x 20 x 20 = batch_size x 2000

    flatten = ad.reshape(activation2, newshape=(batch_size, 2000))

    W3 = ad.Parameter(name='W3', init=rand.normal(scale=0.1, size=(2000, 500)))
    b3 = ad.Parameter(name='b3', init=rand.normal(scale=0.1, size=500))
    Z3 = ad.matmul(flatten, W3)
    Z3 = Z3 + ad.broadcast_to(b3, Z3)
    activation3 = au.nn.relu(Z3)

    W4 = ad.Parameter(name='W4', init=rand.normal(scale=0.1, size=(500, 10)))
    b4 = ad.Parameter(name='b4', init=rand.normal(scale=0.1, size=10))
    logits = ad.matmul(activation3, W4)
    logits = logits + ad.broadcast_to(b4, logits)
    loss = au.nn.cross_entropy_with_logits(logits, y)

    return loss, W1, W2, W3, W4, logits


def measure_accuracy(activation, data, batch_size=32, use_gpu=False):
    X_val, y_val = data

    executor = ad.Executor([activation], use_gpu=use_gpu)

    max_val = len(X_val) - len(X_val) % batch_size
    y_val = y_val[0:max_val]

    prediction = np.zeros(max_val)
    for i in range(0, max_val, batch_size):
        start = i
        end = i + batch_size

        X_batch, y_batch = X_val[start:end], y_val[start:end]
        prob_val, = executor.run(feed_shapes={images: X_batch})

        if use_gpu:
            prob_val = prob_val.asnumpy()
        prediction[start:end] = np.argmax(prob_val, axis=1)

    correct = np.sum(np.equal(y_val, prediction))
    percentage = (correct / len(prediction)) * 100.00
    return percentage


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

    data = au.datasets.MNIST(batch_size=64)
    batch_generator = data.train_batch_generator()

    # images in (batch_size, color_depth, height, width)
    images = ad.Variable(name='images')
    labels = ad.Variable(name='y')

    loss, W1, W2, W3, W4, logits = build_network(images, labels, batch_size=64)
    optimizer = au.optim.Adam(loss, params=[W1, W2, W3, W4], lr=1e-4, use_gpu=use_gpu)

    for i in range(n_iter):
        X_batch, y_batch = next(batch_generator)
        loss_now = optimizer.step(feed_dict={images: X_batch, labels: y_batch})
        if i <= 10 or (i <= 100 and i % 10 == 0) or (i <= 1000 and i % 100 == 0) or (i <= 10000 and i % 500 == 0):
            fmt_str = 'iter: {0:>5d} cost: {1:>8.5f}'
            print(fmt_str.format(i, loss_now[0]))

    # printing validation accuracy
    val_acc = measure_accuracy(logits, data.validation(), batch_size=64, use_gpu=use_gpu)
    print('Validation accuracy: {:>.2f}'.format(val_acc))

    # printing testing accuracy
    test_acc = measure_accuracy(logits, data.testing(), batch_size=64, use_gpu=use_gpu)
    print('Testing accuracy: {:>.2f}'.format(test_acc))

    end = timeit.default_timer()
    print('Time taken for training/testing: {0:.3f} seconds'.format(end - start))
