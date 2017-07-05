# coding: utf-8
# pylint: disable=invalid-name
""" ctypes library of dlsys and helper functions """
from __future__ import absolute_import

import os
import ctypes


def _load_lib():
    """Load libary in build/lib."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    print(curr_path)
    lib_path = os.path.join(curr_path, '../../cuda/build/lib/')
    path_to_so_file = os.path.join(lib_path, "libc_runtime_api.so")
    lib = ctypes.CDLL(path_to_so_file, ctypes.RTLD_GLOBAL)
    return lib


# global library instance
_LIB = _load_lib()


##################
# Helper Methods #
##################

def check_call(ret):
    """Check the return value of C API call

    This function will crash when error occurs.
    Wrap every API call with this function

    Parameters
    ----------
    ret : int
        return value from API calls
    """
    assert(ret == 0)


def c_array(ctype, values):
    """Create ctypes array from a python array

    Parameters
    ----------
    ctype : ctypes data type
        data type of the array we want to convert to

    values : tuple or list
        data content

    Returns
    -------
    out : ctypes array
        Created ctypes array
    """
    return (ctype * len(values))(*values)
