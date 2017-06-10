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
    for n in node.input:
        depth_first_search(n, visited, topo_order)
    topo_order.append(node)
