import numpy as np


def find_topo_sort(node_list):
    """

    :param node_list:
    :return:
    """
    visited = set()
    topo_order = []
    for node in node_list:
        depth_first_search(node, visited, topo_order)
    return topo_order


def depth_first_search(node, visited, topo_order):
    """

    :param node:
    :param visited:
    :param topo_order:
    :return:
    """
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        depth_first_search(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)


def softmax_func(x):
    stable_values = x - np.max(x, axis=1, keepdims=True)
    return np.exp(stable_values) / np.sum(np.exp(stable_values),  axis=1, keepdims=True)
