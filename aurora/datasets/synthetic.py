import numpy as np

# TODO: (upul) improve the interface of following method
def spiral(num_cls, dim, point_per_cls, rnd_state=1024):
    np.random.seed(rnd_state)
    points_per_cls = 100  # number of points per class
    dim = 2  # dimensionality
    num_cls = 3  # number of classes
    X_data = np.zeros((points_per_cls * num_cls, dim))
    y_data = np.zeros(points_per_cls * num_cls, dtype='uint8')
    for j in range(num_cls):
        ix = range(points_per_cls * j, points_per_cls * (j + 1))
        r = np.linspace(0.0, 1, points_per_cls)
        t = np.linspace(j * 4, (j + 1) * 4, points_per_cls) + np.random.randn(points_per_cls) * 0.2  # theta
        X_data[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y_data[ix] = j

    y_data_encoded = np.zeros((points_per_cls * num_cls, num_cls))
    y_data_encoded[range(points_per_cls * num_cls), y_data] = 1
    return X_data, y_data, y_data_encoded
