import numpy as np
import aurora as au
import aurora.autodiff as ad
import timeit
import argparse


def measure_accuracy(activation, data, use_gpu=False):
    X_val, y_val = data
    executor = ad.Executor([activation], use_gpu=use_gpu)
    prob_val, = executor.run(feed_shapes={X: X_val})
    if use_gpu:
        prob_val = prob_val.asnumpy()

    correct = np.sum(np.equal(y_val, np.argmax(prob_val, axis=1)))
    percentage = (correct / (y_val.shape[0])) * 100.00
    return percentage


def build_graph(X, y, input_size, hid_1_size, hid_2_size, output_size):
    # Parameter of the model
    rand = np.random.RandomState(seed=123)
    W1 = ad.Parameter(name="W1", init=rand.normal(scale=0.1, size=(input_size, hid_1_size)))
    b1 = ad.Parameter(name="b1", init=rand.normal(scale=0.1, size=(hid_1_size)))

    W2 = ad.Parameter(name="W2", init=rand.normal(scale=0.1, size=(hid_1_size, hid_2_size)))
    b2 = ad.Parameter(name="b2", init=rand.normal(scale=0.1, size=(hid_2_size)))

    W3 = ad.Parameter(name="W3", init=rand.normal(scale=0.1, size=(hid_2_size, output_size)))
    b3 = ad.Parameter(name="b3", init=rand.normal(scale=0.1, size=(output_size)))

    # building the NN model
    z1 = ad.matmul(X, W1)
    hidden_1 = z1 + ad.broadcast_to(b1, z1)
    activation_1 = au.nn.relu(hidden_1)

    z2 = ad.matmul(activation_1, W2)
    hidden_2 = z2 + ad.broadcast_to(b2, z2)
    activation_2 = au.nn.relu(hidden_2)

    z3 = ad.matmul(activation_2, W3)
    hidden_3 = z3 + ad.broadcast_to(b3, z3)
    loss = au.nn.cross_entropy_with_logits(hidden_3, y)
    return loss, W1, b1, W2, b2, W3, b3, hidden_3


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
    # Create an instance of MNIST dataset and
    # create a generator for reading training data
    data = au.datasets.MNIST(batch_size=128)
    batch_generator = data.training_batch_generator()

    input_size = data.num_features()  # number of features
    hid_1_size = 256  # size of first hidden layer
    hid_2_size = 100  # size of the second hidden layer
    output_size = 10  # size of the output layer

    lr = 1e-3  # learning rate

    # X and y will be used to input data
    X = ad.Variable(name="X")
    y = ad.Variable(name='y')

    loss, W1, b1, W2, b2, W3, b3, logit = build_graph(X, y, input_size, hid_1_size, hid_2_size, output_size)
    # Using Adam optimizer
    # optimizer = au.optim.Adam(loss, params=[W1, b1, W2, b2, W3, b3], lr=lr)
    optimizer = au.optim.SGD(loss, params=[W1, b1, W2, b2, W3, b3], lr=lr, use_gpu=use_gpu)
    # Starts training
    for i in range(n_iter):
        # read next random batch from the training generator
        X_batch, y_batch = next(batch_generator)
        # run the optimizer and it will return the cost
        # after that iteration
        loss_now = optimizer.step(feed_dict={X: X_batch, y: y_batch})
        if i <= 10 or (i <= 100 and i % 10 == 0) or (i <= 1000 and i % 100 == 0) or (i <= 10000 and i % 500 == 0):
            fmt_str = 'iter: {0:>5d} cost: {1:>8.5f}'
            print(fmt_str.format(i, loss_now[0]))

    # printing validation accuracy
    # TODO (upul) optimize hyper-parameters using validation dataset
    val_acc = measure_accuracy(logit, data.validation(), use_gpu=use_gpu)
    print('Validation accuracy: {:>.2f}'.format(val_acc))

    # printing testing accuracy
    test_acc = measure_accuracy(logit, data.testing(), use_gpu=use_gpu)
    print('Testing accuracy: {:>.2f}'.format(test_acc))

    end = timeit.default_timer()
    print('\nTime taken for training/testing: {0:3f}'.format(end - start))
