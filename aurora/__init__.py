import aurora.nn
import aurora.optim
import aurora.datasets
from config import sys_configs

__all__ = ["nn", "optim", "datasets"]

if sys_configs['use_gpu']:
    from aurora.ndarray import ndarray, gpu_op

    __all__ = __all__.append("ndarray")
