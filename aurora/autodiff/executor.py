from .autodiff import PlaceholderOp
from .utils import find_topo_sort


class Executor:
    """

    """

    def __init__(self, eval_list, use_gpu=False):
        """
        Executor computes values for a given subset of nodes in a computation graph.

        Parameters:
        -----------
        :param eval_list: Values of the nodes of this list need to be computed
        """
        self.eval_list = eval_list
        self.use_gpu = use_gpu
        self.node_to_arr_map = {}
        self.node_to_shape_map = {}

    def infer_shape(self, feed_shapes):
        """
        Given the shapes of the feed_shapes dictionary, we infer shapes of all nodes in the graph
        :param feed_shapes:
        :return:
        """
        self.node_to_shape_map = {}
        topo_order = find_topo_sort(self.eval_list)  # TODO (upul) cache this
        for node in topo_order:
            if node in self.node_to_shape_map:
                continue
            if node in feed_shapes:
                self.node_to_shape_map[node] = feed_shapes[node]
            else:
                input_shapes = []
                for input_node in node.inputs:  # TODO: (upul) list comprehension
                    input_shapes.append(self.node_to_shape_map[node])
                self.node_to_shape_map[node] = node.op.infer_shape(input_shapes)

    def memory_plan(self, feed_shapes):
        """

        :param feed_shapes:
        :return:
        """
        topo_order = find_topo_sort(self.eval_list)  # TODO (upul) cache this
        self.node_to_arr_map = {}
        for node in topo_order:
            pass
            # self.node_to_arr_map[node] = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)

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

            # TODO (upul): following if condition looks like a hack. Find a better approach
            if isinstance(node.op, PlaceholderOp) and node.const is not None:
                node_to_eval_map[node] = node.const
                continue

            inputs = [node_to_eval_map[n] for n in node.inputs]
            value = node.op.compute(node, inputs)
            node_to_eval_map[node] = value

        # select values of nodes given in feed_dicts
        return [node_to_eval_map[node] for node in self.eval_list]
