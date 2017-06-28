from aurora.autodiff.autodiff import Op


class Conv2dOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = 'Conv2d({0:s}, {1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        pass

    def gradient(self, node, output_grads):
        pass
