# Aurora: Minimal Deep Learning Library.

Aurora is a minimal deep learning library written in Python/Numpy and a bit of C++. It was designed to construct simple deep learning systems such as simple MLP. The current version comes with following features.

* Automatic differentiation using static computational graphs.
* Shape inference.
* Static memory allocation for efficient training and inference.
* Support both GPU (using Nvidia CUDA) and numpy.

Tough Aurora in a minimal deep learning system, it is quite capable of building MLPs for real-world datasets such as MINST and CIFAR-10. 

### Limitations

The current version comes with following limitations. We will be addressing those limitations in upcoming releases.

* Convolutional operators.
* cuDNN support.
* Model checkpointing.
* Multi-GPU and distributed training.

### How to Install

### Examples

```python
import numpy as np
import aurora as au
import aurora.autodiff as ad
import matplotlib.pyplot as plt

lr = 1e-3
n_epoch = 100
num_point = 250

x = ad.Variable(name='x')
y = ad.Variable(name='y')
W = ad.Variable(name='W')
b = ad.Variable(name='b')
z = ad.matmul(x, W)
output = z + ad.broadcast_to(b, z)

cost = ad.reduce_sum((y - output) * (y - output)) / (2.0 * num_point)
grad_cost_w, grad_b = ad.gradients(cost, [W, b])

x_data = np.linspace(0, 5, num_point).reshape((num_point, 1))
y_data = 2.0 * x_data + np.random.uniform(-1.0, 1.0, (num_point, 1)) + 2.5 * np.ones((num_point, 1))

w_val = np.zeros((1, 1))
b_val = np.zeros(1)
optimizer = au.optim.SGD(cost, params=[W, b], lr=lr, use_gpu=True)

for i in range(n_epoch):
    # evaluate the graph
    cost_now = optimizer.step(feed_dict={X: x_data, y: y_data})
    fmt_str = 'iter: {0:>5d} cost: {1:>8.5f}'
    print(fmt_str.format(i, cost_now[0]))
```

### References.

