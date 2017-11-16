import numpy as np

from aurora.autodiff.autodiff import PlaceholderOp
from aurora.ndarray import ndarray
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
        self.eval_node_list = eval_list
        self.ctx = None
        if use_gpu:
            self.ctx = ndarray.gpu(0)

        self.topo_order = find_topo_sort(self.eval_node_list)
        self.node_to_arr_map = None
        self.node_to_shape_map = None
        self.feed_shapes = None

    def infer_shape(self, feed_shapes):
        """
        Given the shapes of the feed_shapes dictionary, we infer shapes of all nodes in the graph
        :param feed_shapes:
        :return:
        """
        self.node_to_shape_map = {}
        for node in self.topo_order:
            if node in self.node_to_shape_map:
                continue

            # TODO (upul): following if condition looks like a hack. Find a better approach
            if isinstance(node.op, PlaceholderOp) and node.const is not None:
                self.node_to_shape_map[node] = node.const.shape
                continue

            if node in feed_shapes:
                self.node_to_shape_map[node] = feed_shapes[node]
            else:
                input_shpes = []
                for input_node in node.inputs:
                    input_shpes.append(self.node_to_shape_map[input_node])

                self.node_to_shape_map[node] = node.op.infer_shape(node, input_shpes)

    def memory_plan(self, feed_shapes):
        """

        :param feed_shapes:
        :return:
        """
        # topo_order = find_topo_sort(self.eval_node_list)  # TODO (upul) cache this
        # self.node_to_arr_map = {}
        # for node in topo_order:
        #     self.node_to_arr_map[node] = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)

        if self.node_to_arr_map is None:
            self.node_to_arr_map = {}

        for node in self.topo_order:
            if node in feed_shapes:
                continue
            self.node_to_arr_map[node] = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)

    def run(self, feed_shapes, convert_to_numpy_ret_vals=False):
        """
        Values of the nodes given in eval_list are evaluated against feed_dict

        Parameters
        ----------
        :param feed_shapes: A dictionary of nodes who values are specified by the user

        Returns
        -------
        :return: Values of the nodes specified by the eval_list
        """
        # node_to_eval_map = dict(feed_dict)
        # topo_order = find_topo_sort(self.eval_list)
        # for node in topo_order:
        #     if node in feed_dict:
        #         continue
        #
        #     # TODO (upul): following if condition looks like a hack. Find a better approach
        #     if isinstance(node.op, PlaceholderOp) and node.const is not None:
        #         node_to_eval_map[node] = node.const
        #         continue
        #
        #     inputs = [node_to_eval_map[n] for n in node.inputs]
        #     value = node.op.compute(node, inputs)
        #     node_to_eval_map[node] = value
        #
        # # select values of nodes given in feed_dicts
        # return [node_to_eval_map[node] for node in self.eval_list]

        #use_numpy = self.ctx is None

        def are_feed_shapes_equal(sa, sb):
            if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
                return False
            unmatched_item = set(sa.items()) ^ set(sb.items())
            return len(unmatched_item) == 0

        # Assume self.ctx is None implies numpy array and numpy ops.
        use_numpy = self.ctx is None
        node_to_val_map = {}
        for node, value in feed_shapes.items():
            if use_numpy:
                # all values passed in feed_dict must be np.ndarray
                assert isinstance(value, np.ndarray)
                node_to_val_map[node] = value
            else:
                # convert values to ndarray.NDArray if necessary
                if isinstance(value, np.ndarray):
                    node_to_val_map[node] = ndarray.array(value, ctx=self.ctx)
                elif isinstance(value, ndarray.NDArray):
                    node_to_val_map[node] = value
                else:
                    assert False, "feed_dict value type not supported"

        # collect shapes for all placeholders
        feed_shapes = {}
        for node in node_to_val_map:
            feed_shapes[node] = node_to_val_map[node].shape

        # infer shape if feed_shapes changed since last run
        # e.g. call run() on test data after trainng
        if (not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            # plan memory if using GPU
            if (not use_numpy):
                self.memory_plan(feed_shapes)

        # Traverse graph in topo order and compute values for all nodes.
        for node in self.topo_order:
            if node in node_to_val_map:
                # Skip placeholder nodes. Values already provided by feed_dict.
                continue

            # TODO (upul): following if condition looks like a hack. Find a better approach
            if isinstance(node.op, PlaceholderOp) and node.const is not None:
                node_to_val_map[node] = node.const
                continue

            input_vals = [node_to_val_map[n] for n in node.inputs]
            if use_numpy:
                node_val = np.empty(shape=self.node_to_shape_map[node])
            else:
                node_val = self.node_to_arr_map[node]
            # node_val is modified in-place whether np.ndarray or NDArray
            node.op.compute(node, input_vals, node_val, use_numpy)
            node_to_val_map[node] = node_val

        # Collect node values.
        if not use_numpy and convert_to_numpy_ret_vals:
            return [node_to_val_map[n].asnumpy() for n in self.eval_node_list]

        return [node_to_val_map[n] for n in self.eval_node_list]

    @staticmethod
    def _are_feed_shapes_equal(sa, sb):
        if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
            return False
        unmatched_items = set(sa.items()) ^ set(sb.items())
        return len(unmatched_items)




# class Executor(object):
#     """Executor computes values for given set of nodes in computation graph."""
#
#     def __init__(self, eval_node_list, use_gpu = False):
#         """
#         Parameters
#         ----------
#         eval_node_list: list of nodes whose values need to be computed.
#         ctx: runtime DLContext, default is None which means np.ndarray on cpu
#         topo_order: list of nodes in topological order
#         node_to_shape_map: dict from node to shape of the node
#         node_to_arr_map: dict from node to ndarray.NDArray allocated for node
#         feed_shapes: shapes of feed_dict from last run(...)
#         """
#         self.eval_node_list = eval_node_list
#         self.ctx = None
#         if use_gpu:
#             self.ctx = ndarray.gpu(0)
#         self.topo_order = find_topo_sort(self.eval_node_list)
#         self.node_to_shape_map = None
#         self.node_to_arr_map = None
#         self.feed_shapes = None
#
#     def infer_shape(self, feed_shapes):
#         """Given shapes of feed_dict nodes, infer shape for all nodes in graph.
#         Implementation note:
#         Iteratively calls node.op.infer_shape to infer shapes.
#         Node shapes stored in self.node_to_shape_map.
#         Parameters
#         ----------
#         feed_shapes: node->shapes mapping for feed_dict nodes.
#         """
#         """TODO: Your code here"""
#         self.node_to_shape_map = {}
#         for node in self.topo_order:
#             if node in self.node_to_shape_map:
#                 continue
#
#             if node in feed_shapes:
#                 self.node_to_shape_map[node] = feed_shapes[node]
#             else:
#                 input_shpes = []
#                 for input_node in node.inputs:
#                     input_shpes.append(self.node_to_shape_map[input_node])
#
#                 self.node_to_shape_map[node] = node.op.infer_shape(node, input_shpes)
#
#     def memory_plan(self, feed_shapes):
#         """Allocates ndarray.NDArray for every node except feed_dict nodes.
#         Implementation note:
#         Option 1: Alloc a ndarray.NDArray per node that persists across run()
#         Option 2: Implement a memory pool to reuse memory for nodes of same
#                 shapes. More details see Lecture 7.
#         For both options, self.node_to_arr_map stores node->NDArray mapping to
#         allow mapping to persist across multiple executor.run().
#         Hint: use ndarray.empty(shape, ctx=self.ctx) to allocate NDArray.
#         Parameters
#         ----------
#         feed_shapes: node->shapes mapping for feed_dict nodes.
#         """
#         """TODO: Your code here"""
#         if self.node_to_arr_map is None:
#             self.node_to_arr_map = {}
#
#         for node in self.topo_order:
#             if node in feed_shapes:
#                 continue
#             self.node_to_arr_map[node] = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)
#
#     def run(self, feed_shapes, convert_to_numpy_ret_vals=False):
#         """
#         Parameters
#         ----------
#         feed_shapes: a dictionary of node->np.ndarray supplied by user.
#         convert_to_numpy_ret_vals: whether to convert ret vals to np.array
#         Returns
#         -------
#         A list of values for nodes in eval_node_list. NDArray or np.ndarray.
#         """
#
#         def are_feed_shapes_equal(sa, sb):
#             if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
#                 return False
#             unmatched_item = set(sa.items()) ^ set(sb.items())
#             return len(unmatched_item) == 0
#
#         # Assume self.ctx is None implies numpy array and numpy ops.
#         use_numpy = self.ctx is None
#         node_to_val_map = {}
#         for node, value in feed_shapes.items():
#             if use_numpy:
#                 # all values passed in feed_dict must be np.ndarray
#                 assert isinstance(value, np.ndarray)
#                 node_to_val_map[node] = value
#             else:
#                 # convert values to ndarray.NDArray if necessary
#                 if isinstance(value, np.ndarray):
#                     node_to_val_map[node] = ndarray.array(value, ctx=self.ctx)
#                 elif isinstance(value, ndarray.NDArray):
#                     node_to_val_map[node] = value
#                 else:
#                     assert False, "feed_dict value type not supported"
#
#         # collect shapes for all placeholders
#         feed_shapes = {}
#         for node in node_to_val_map:
#             feed_shapes[node] = node_to_val_map[node].shape
#
#         # infer shape if feed_shapes changed since last run
#         # e.g. call run() on test data after trainng
#         if (not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
#             self.infer_shape(feed_shapes)
#             self.feed_shapes = feed_shapes
#             # plan memory if using GPU
#             if (not use_numpy):
#                 self.memory_plan(feed_shapes)
#
#         # Traverse graph in topo order and compute values for all nodes.
#         for node in self.topo_order:
#             if node in node_to_val_map:
#                 # Skip placeholder nodes. Values already provided by feed_dict.
#                 continue
#             input_vals = [node_to_val_map[n] for n in node.inputs]
#             if use_numpy:
#                 node_val = np.empty(shape=self.node_to_shape_map[node])
#             else:
#                 node_val = self.node_to_arr_map[node]
#             # node_val is modified in-place whether np.ndarray or NDArray
#             node.op.compute(node, input_vals, node_val, use_numpy)
#             node_to_val_map[node] = node_val
#
#         # Collect node values.
#         if not use_numpy and convert_to_numpy_ret_vals:
#             return [node_to_val_map[n].asnumpy() for n in self.eval_node_list]
#         return [node_to_val_map[n] for n in self.eval_node_list]