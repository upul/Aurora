import numpy as np
from .helper import find_topo_sort
from .helper import sum_node_list


class Node(object):
    """ Node object represents a node in the computational graph"""

    def __init__(self):
        """ New node will be created by Op objects __call__ method"""
        # list of inputs to this node
        self.inputs = []
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

    def __sub__(self, other):
        if isinstance(other, Node):
            return sub(self, other)
        else:
            return sub_const(self, other)

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            return mul_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            return div_const(self, other)

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rdiv__ = __truediv__


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
        new_node.inputs = [node_A]
        new_node.name = '({0:s}+{1:f})'.format(node_A.name, const_val)
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


class SubOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = '({0:s}-{1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] - input_vals[1]

    def gradient(self, node, output_grads):
        return [output_grads, -1 * output_grads]


class SubByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const = const_val
        new_node.name = '({0:s}-{1:f})'.format(node_A.name, const_val)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] - node.const

    def gradient(self, node, output_grads):
        return [output_grads]


class OnesLikeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Oneslike({})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert isinstance(input_vals[0], np.ndarray)
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grads):
        return [zeros_like(node.inputs[0])]


class ZerosLikeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'Zeroslike({})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert isinstance(input_vals[0], np.ndarray)
        return np.zeros_like(input_vals[0].shape)

    def gradient(self, node, output_grads):
        return [zeros_like(node.inputs[0])]


class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = '({0:s}*{1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grads):
        return [node.inputs[1] * output_grads, node.inputs[0] * output_grads]


class MulByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const = const_val
        new_node.name = '({0:s}*{1:f})'.format(node_A.name, const_val)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return node.const * input_vals[0]

    def gradient(self, node, output_grads):
        return [node.const * output_grads]


class DivOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = '({0:s}/{1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] / input_vals[1]

    def gradient(self, node, output_grads):
        return [output_grads / node.inputs[1], -1.0 * output_grads * node.inputs[0] / (node.inputs[1] * node.inputs[1])]


class DivByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const = const_val
        new_node.name = '({0:s}/{1:f})'.format(node_A.name, const_val)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] / node.const

    def gradient(self, node, output_grads):
        return [output_grads / node.const]


class PlaceholderOp(Op):
    """Op to feed value to a nodes."""

    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None


class ReduceSumOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = 'ReduceSum({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.sum(input_vals[0], axis=0)

    def gradient(self, node, output_grads):
        return [output_grads]


def Variable(name):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder()
    placeholder_node.name = name
    return placeholder_node


# Global singleton operations
add = AddOp()
add_const = AddByConstOp()
sub = SubOp()
sub_const = SubByConstOp()
mul = MulOp()
mul_const = MulByConstOp()
div = DivOp()
div_const = DivByConstOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
placeholder = PlaceholderOp()
reduce_sum = ReduceSumOp()


def gradients(output_node, node_list):
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list = {}
    # Special note on initializing gradient of output_node as oneslike_op(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_node] = [ones_like(output_node)]
    # a map from node to the gradient of that node
    node_to_output_grad = {}
    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    """TODO: Your code here"""
    for node in reverse_topo_order:
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad

        input_grads_list = node.op.gradient(node, output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            node_to_output_grads_list[node.inputs[i]].append(input_grads_list[i])

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list


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
            inputs = [node_to_eval_map[n] for n in node.inputs]
            value = node.op.compute(node, inputs)
            node_to_eval_map[node] = value

        # select values of nodes given in feed_dict
        return [node_to_eval_map[node] for node in self.eval_list]
