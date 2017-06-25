from .autodiff import PlaceholderOp
from .utils import find_topo_sort


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

            # TODO: following if condition looks like a hack. Find a better approach
            if isinstance(node.op, PlaceholderOp) and node.const is not None:
                node_to_eval_map[node] = node.const
                continue

            inputs = [node_to_eval_map[n] for n in node.inputs]
            value = node.op.compute(node, inputs)
            node_to_eval_map[node] = value

        # select values of nodes given in feed_dicts
        return [node_to_eval_map[node] for node in self.eval_list]
