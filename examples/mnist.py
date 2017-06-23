import numpy as np
import aurora.autodiff as ad
from aurora.optim import SGD

# TODO: loading training/testing data
# TODO: set dimensions

X = ad.Variable(name="X")
y = ad.Variable(name='y')

W1 = ad.Parameter(name="W1", init=0.01 * np.random.randn(, ))
b1 = ad.Parameter(name="b1", init=np.zeros())

W2 = ad.Parameter(name="W2", init=0.01 * np.random.randn(, ))
b2 = ad.Parameter(name="b2", init=np.zeros())

W3 = ad.Parameter(name="W3", init=0.01 * np.random.randn(, ))
b3 = ad.Parameter(name="b3", init=np.zeros())

z1 = ad.matmul(X, W1)
hidden_1 = z1 + ad.broadcast_to(b1, z1)
activation_1 = ad.relu(hidden_1)

z2 = ad.matmul(activation_1, W2)
hidden_2 = z2 + ad.broadcast_to(b2, z2)
activation_2 = ad.relu(hidden_2)

z3 = ad.matmul(activation_2, W3)
hidden_3 = z3 + ad.broadcast_to(b3, z3)
loss = ad.cross_entropy(hidden_3, y)
