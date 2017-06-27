import numpy as np

def softmax_func(x):
    stable_values = x - np.max(x, axis=1, keepdims=True)
    return np.exp(stable_values) / np.sum(np.exp(stable_values),  axis=1, keepdims=True)