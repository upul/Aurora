<!--p align="center">
    <img src="https://github.com/upul/Aurora/blob/master/resources/logo.png" alt="logo">
</p-->
# Aurora: Minimal Deep Learning Library.

Aurora is a minimal deep learning library written in Python/Numpy and a bit of C++. It was designed to construct simple deep learning systems such as simple MLP. The current version comes with following features.

* Automatic differentiation using static computational graphs.
* Shape inference.
* Static memory allocation for efficient training and inference.
* Support both GPU (using Nvidia CUDA) and numpy.

Tough Aurora in a minimal deep learning system, it is quite capable of building MLPs for real-world datasets such as MINST and CIFAR-10. 

### Future Work

Following features will be added in upcoming releases.

* Dropout and Batch-norm.
* Convolutional operators.
* cuDNN support.
* Model checkpointing.
* Multi-GPU and distributed training.

### Installation

Aurora relies on number of external libraries including CUDA and NumPy. For CUDA installation instruction please refer official CUDA documentation. Python dependencies can be install by running `requirements.txt` file.

##### Environment setup
In order to use GPU capabilities of the Aurora library, you need to have a Nvidia GPU. If CUDA toolkit is not already installed, first install the latest version of the CUDA toolkit. Next, set following environment variables.

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
```

##### Cloning the Repository

You can clone Aurora repository using following command.

`git clone https://github.com/upul/Aurora.git`


##### Building the GPU Backend

Next, you need to build GPU backend. So please `cuda` directory and run `make` command as shown below.

1. Go to `cuda` directory `cd cuda`
2. Run `make`

##### Installing the Library

Go to `Aurora` directory and run:

1. `pip install -r requirements.txt`
2. `pip install .`

### Examples

Following are some of the examples written in Aurora. For complete list of examples please refer [`examples`](https://github.com/upul/Aurora/tree/master/examples) directory. Also, we have created few `Jupyter` notebooks and please refer [`examples/notebooks`](https://github.com/upul/Aurora/tree/master/examples/notebooks) for details. 

1. [Linear Regression](https://github.com/upul/Aurora/blob/master/examples/linear_regression_optim.py)
2. [Softmax](https://github.com/upul/Aurora/blob/master/examples/softmax.py)
3. [Toy Neural Network](https://github.com/upul/Aurora/blob/master/examples/toy_neural_network.py)
4. [MNIST](https://github.com/upul/Aurora/blob/master/examples/mnist.py)


### References.

1. 
2. 
