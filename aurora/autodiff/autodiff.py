import numpy as np
from .helper import find_topo_sort

class Node(object):
    """ Node object represents a node in the computational graph"""

    def __init__(self):
        """ New node will be created by Op objects __call__ method"""
        # list of inputs to this node
        self.input = []
        # operator
        self.op = None
        # constants
        self.const = None
        # name of the node mainly use for debugging
        self.name = ""

    def __add__(self, other):
        """ Adding two nodes and returns a new node"""
        if isinstance(other, Node):
            return add(self, other)
        else:
            return add_const(self, other)


class Op(object):
    """ Op class represents operations perform on nodes"""

    def __call__(self):
        """
        Create a new node which represents operations perform on the graph

        Parameters
        ----------
        None

        Returns
        -------
        Node
            The new node object
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        """
        Given the values of input nodes, compute the output value

        Parameters
        ----------
        :param node: Node that performs the computation
        :param input_vals: Values of input node

        Returns
        -------
        :return: The output value of the node
        """
        raise NotImplementedError

    def gradient(self, node, output_grads):
        """
        Given the value of output gradients this operation calculate the
        gradient contribution of each input node

        Parameters
        ----------
        :param node:
        :param output_grads:

        Returns
        -------
        :return: A list of gradient contribution to each input node respectively
        """
        raise NotImplementedError


class AddOp(Op):
    """

    """

    def __call__(self, nodeA, nodeB):
        """
        This Operator element-wise two nodes

        Parameters
        ----------
        :param nodeA: LHS operand
        :param nodeB: RHS operand

        Returns
        -------
        :return: A new Node which represents the element-wise plus operation
        """
        new_node = Op.__call__(self)
        new_node.inputs = [nodeA, nodeB]
        new_node.name = '({}+{})'.format(nodeA.name, nodeB.name)
        return new_node

    def compute(self, node, input_vals):
        """
        Given values of two input nodes, return result of element-wise addition.
        Parameters
        ----------
        :param node:
        :param input_vals: List of two input nodes

        Returens
        --------
        :return:  The result of the element-wise addition operation
        """
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grads):
        """
        Given the values of output gradients, calculate the gradients of input nodes

        Parameters
        ----------
        :param node:
        :param output_grads: Gradient contribution of output nodes

        Returns
        -------
        :return: A list of gradient contribution of output nodes
        """
        return [output_grads[0], output_grads[1]]


class AddByConstOp(Op):
    """
    Operator represents the element-wise addition of a node and a const
    """

    def __call__(self, node_A, const_val):
        """

        :param node:
        :param const_val:
        :return:
        """
        new_node = Op.__call__(self)
        new_node.const = const_val
        new_node.input = [node_A]
        new_node.name = '({s}+{s})'.format(node_A.name, const_val)
        return new_node

    def compute(self, node, input_vals):
        """

        :param node:
        :param input_vals:
        :return:
        """
        assert len(input_vals) == 1
        return node.const + input_vals[0]

    def gradient(self, node, output_grads):
        """

        :param node:
        :param output_grads:
        :return:
        """
        return [output_grads]

# Global singleton operations
add = AddOp()
add_const = AddByConstOp()


class Executor:
    """

    """

    def __init__(self, eval_list):
        """
        Executor computes values for a given subset of nodes in a computation graph.
        
        Parameters:
        -----------
        :param eval_list: Values of the nodes of this list need to be computed
        """
        self.eval_list = eval_list

    def run(self, feed_dict):
        """
        Values of the nodes given in eval_list are evaluated against feed_dict

        Parameters
        ----------
        :param feed_dict: A dictionary of nodes who values are specified by the user

        Returns
        -------
        :return: Values of the nodes specified by the eval_list
        """
        node_to_eval_map = dict(feed_dict)
        topo_order = find_topo_sort(self.eval_list)
        for node in topo_order:
            if node in feed_dict:
                continue
            inputs = [node_to_eval_map[n] for n in node.input]
            value = node.op.compute(inputs)
            node_to_eval_map[node] = value

        # select values of nodes given in feed_dict
        return [node_to_eval_map[node] for node in self.eval_list]

