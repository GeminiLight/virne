from .network import Network


class VirtualNetwork(Network):
    def __init__(self, incoming_graph_data=None, node_attrs_setting=[], edge_attrs_setting=[], **kwargs):
        super(VirtualNetwork, self).__init__(incoming_graph_data, node_attrs_setting, edge_attrs_setting, **kwargs)

    def generate_topology(self, num_nodes, type='random', **kwargs):
        return super().generate_topology(num_nodes, type=type, **kwargs)


if __name__ == '__main__':
    pass