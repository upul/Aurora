# Aurora: Minimal Deep Learning Library.

Aurora is a minimal deep learning library written in Python, Cython, and C++ with the help of Numpy, CUDA, and cuDNN. Though it is simple, Aurora comes with some advanced design concepts found it a typical deep learning library. 

* Automatic differentiation using static computational graphs.
* Shape and type inference.
* Static memory allocation for efficient training and inference.


### Installation

Aurora relies on several external libraries including `CUDA`, `cuDNN`, and `NumPy`. For `CUDA` and `cuDNN` installation instructions please refer official documentation. Python dependencies can be installed by running the `requirements.txt` file.

##### Environment setup

To utilize GPU capabilities of the Aurora library, you need to have a Nvidia GPU. If `CUDA` toolkit is not already installed, first install the latest version of the `CUDA` toolkit as well as `cuDNN` library. Next, set following environment variables.

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
```

##### Cloning the Repository

You can clone Aurora repository using following command.

`git clone https://github.com/upul/Aurora.git`


##### Building the GPU Backend

Next, you need to build GPU backend. So please `cuda` directory and run `make` command as shown below.

1. Go to `cuda` directory (`cd cuda`)
2. Run `make`

##### Installing the Library

Go to `Aurora` directory and run:

1. `pip install -r requirements.txt`
2. `pip install .`


### Examples

Following lists some noticeable examples. For the complete list of examples please refer [`examples`](https://github.com/upul/Aurora/tree/master/examples) directory. Also,  for Jupyter notebooks please refer [`examples/notebooks`](https://github.com/upul/Aurora/tree/master/examples/notebooks) folder.

1. [mnist](https://github.com/upul/Aurora/blob/master/examples/mnist.py)
2. [mnist_cnn](https://github.com/upul/Aurora/blob/master/examples/mnist_cnn.py)


### Future Work

Following features will be added in upcoming releases.

* Dropout and Batch Normalization.
* High-level API similar to Keras.
* Ability to load pre-trained models.
* Model checkpointing.


### Acknowledgement

It all started with [CSE 599G1: Deep Learning System Design](http://dlsys.cs.washington.edu/) course. This course really helped me to understand fundamentals of Deep Learning System design. My answers to the two programming assignments of [CSE 599G1](http://dlsys.cs.washington.edu/) was the foundation of Aurora library.  So I would like to acknowledge with much appreciation the instructors and teaching assistants of the  [SE 599G1](http://dlsys.cs.washington.edu/) course.


### References.

1. [CSE 599G1: Deep Learning System Design](http://dlsys.cs.washington.edu/) 
2. [MXNet Architecture](https://mxnet.incubator.apache.org/architecture/index.html)
3. [Parallel Programming With CUDA | Udacity](https://www.udacity.com/course/intro-to-parallel-programming--cs344)
4. [Programming Massively Parallel Processors, Third Edition: A Hands-on Approach 3rd Edition](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0128119861/ref=pd_sim_14_3?_encoding=UTF8&psc=1&refRID=1Z3KFKEPTFQJE7MZQ40G)
