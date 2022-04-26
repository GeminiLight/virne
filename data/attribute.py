import copy
import numpy as np
import networkx as nx

from .utils import path_to_edges, generate_data_with_distribution


"""
To-do
Energy: Node Energy = (pmnode - pbnode)*  (snode.cpu - snode.lastcpu) / snode.cpu + pbnode
"""


class Attribute(object):

    def __init__(self, name, owner, type, *args, **kwargs):
        self.name = name
        self.owner = owner
        self.type = type

        # for extrema
        if type == 'extrema':
            self.originator = kwargs.get('originator')

        # for generative
        self.generative = kwargs.get('generative', False)

        if self.generative:
            self.distribution = kwargs.get('distribution', 'normal')
            self.dtype = kwargs.get('dtype', 'float')
            assert self.distribution in ['normal', 'uniform', 'exponential', 'possion', 'customized']
            assert self.dtype in ['int', 'float', 'bool']
            if self.distribution in ['uniform']:
                self.low = kwargs.get('low', 0.)
                self.high = kwargs.get('high', 1.)
            elif self.distribution in ['normal']:
                self.loc = kwargs.get('loc', 0.)
                self.scale = kwargs.get('scale', 1.)
            elif self.distribution in ['exponential']:
                self.scale = kwargs.get('scale', 1.)
            elif self.distribution in ['possion']:
                self.scale = kwargs.get('lam', 1.)
            elif self.distribution in ['customized']:
                self.min = kwargs.get('low', 0.)
                self.max = kwargs.get('high', 1.)

    @classmethod
    def from_dict(cls, dict):
        dict_copy = copy.deepcopy(dict)
        name = dict_copy.pop('name')
        owner = dict_copy.pop('owner')
        type = dict_copy.pop('type')
        assert (owner, type) in ATTRIBUTES_DICT.keys(), ValueError('Unsupproted attribute!')
        AttributeClass = ATTRIBUTES_DICT.get((owner, type))
        return AttributeClass(name, **dict_copy)

    def check(self, *args, **kwargs):
        return True

    def _generate_data_with_dist(self, network):
        assert self.generative
        size = network.num_nodes if self.owner == 'node' else network.num_edges
        if self.distribution == 'uniform':
            kwargs = {'low': self.low, 'high': self.high}
        elif self.distribution == 'normal':
            kwargs = {'loc': self.loc, 'scale': self.scale}
        elif self.distribution == 'exponential':
            kwargs = {'scale': self.scale}
        elif self.distribution == 'possion':
            kwargs = {'lam': self.lam}
        else:
            raise NotImplementedError
        return generate_data_with_distribution(size, distribution='uniform', dtype=self.dtype, **kwargs)

    def to_dict(self, ):
        return self.__dict__

    def __repr__(self):
        info = [f'{key}={self._size_repr(item)}' for key, item in self.__dict__]
        return f"{self.__class__.__name__}({', '.join(info)})"


class InfoAttribute(Attribute):

    def __init__(self, name, owner, *args, **kwargs):
        super().__init__(name, owner, 'info', *args, **kwargs)


class NodeInfoAttribute(InfoAttribute):

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, 'node', *args, **kwargs)

class EdgeInfoAttribute(InfoAttribute):

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, 'edge', *args, **kwargs)


### Public Methods ###
class ResourceMethod:

    def update(self, v, p, method='+'):
        assert self.type in ['resource']
        assert method in ['+', '-', 'add', 'sub']
        if method in ['+', 'add']:
            p[self.name] += v[self.name]
        elif method in ['-', 'sub']:
            p[self.name] -= v[self.name]
            assert p[self.name] >= 0
        else:
            raise NotImplementedError
        return True
    
    def _check_one_element(self, v, p, method='ge'):
        assert method in ['>=', '<=', 'ge', 'le']
        if method in ['>=', 'ge']:
            return p[self.name] >= v[self.name]
        elif method == ['-', 'sub']:
            return p[self.name] <= v[self.name]
        else:
            raise NotImplementedError

    def generate_data(self, network):
        if self.generative:  # generative attribute
            return self._generate_data_with_dist(network)
        else:
            raise NotImplementedError

class ExtremaMethod:

    def update(self, v, p, method='+'):
        return True

    def check(self, vn, pn, v_node_id, p_node_id, method='ge'):
        return True

    def generate_data(self, network):
        if self.owner == 'node':
            originator_attribute = network.node_attrs[self.originator]
        else:
            originator_attribute = network.edge_attrs[self.originator]
        attribute_data = originator_attribute.get_data(network)
        return attribute_data


class NodeMethod:

    def set_data(self, network, attribute_data):
        if not isinstance(attribute_data, dict):
            attribute_data = {n: attribute_data[i] for i, n in enumerate(network.nodes)}
        nx.set_node_attributes(network, attribute_data, self.name)

    def get_data(self, network):
        attribute_data = list(nx.get_node_attributes(network, self.name).values())
        return attribute_data


class EdgeMethod:

    def set_data(self, network, attribute_data):
        if not isinstance(attribute_data, dict):
            attribute_data = {e: attribute_data[i] for i, e in enumerate(network.edges)}
        nx.set_edge_attributes(network, attribute_data, self.name)

    def get_data(self, network):
        attribute_data = list(nx.get_edge_attributes(network, self.name).values())
        return attribute_data

    def get_adjacency_data(self, network, normalized=False):
        adjacency_data = nx.attr_sparse_matrix(
            network, edge_attr=self.name, normalized=normalized, rc_order=network.nodes).T
        return adjacency_data

    def get_aggregation_data(self, network, aggr='sum', normalized=False):
        assert aggr in ['sum', 'mean'], NotImplementedError
        aggregation_data = nx.attr_sparse_matrix(
                network, edge_attr=self.name, normalized=normalized, rc_order=network.nodes)
        if aggr == 'sum':
            aggregation_data = aggregation_data.sum(axis=0)
        elif aggr == 'mean':
            aggregation_data = aggregation_data.mean(axis=0)
        aggregation_data = np.asarray(aggregation_data)[0]
        return aggregation_data


# Node Attributes
class NodeResourceAttribute(Attribute, NodeMethod, ResourceMethod):
    def __init__(self, name, *args, **kwargs):
        super(NodeResourceAttribute, self).__init__(name, 'node', 'resource', *args, **kwargs)

    def check(self, vn, pn, v_node_id, p_node_id, method='ge'):
        v, p = vn.nodes[v_node_id], pn.nodes[p_node_id]
        return super()._check_one_element(v, p, method)

class NodeExtremaAttribute(Attribute, NodeMethod, ExtremaMethod):
    def __init__(self, name, *args, **kwargs):
        super(NodeExtremaAttribute, self).__init__(name, 'node', 'extrema', *args, **kwargs)


class NodePositionAttribute(Attribute, NodeMethod):

    def __init__(self, name='pos', *args, **kwargs):
        super(NodePositionAttribute, self).__init__(name, 'node', 'position', *args, **kwargs)

    def generate_data(self, network):
        if self.generative:
            pos_x = self._generate_data_with_dist(network)
            pos_y = self._generate_data_with_dist(network)
            pos_data = [(x, y) for x, y in zip(pos_x, pos_y)]
        elif 'pos' in network.nodes[0].keys():
            pos_data = list(nx.get_node_attributes(network, 'pos').values())
        else:
            return AttributeError('Please specify how to generate data')
        return pos_data

    def check(self, vn, pn, vnf_id, pn_node_id, **kwargs):
        return True

# Edge Attributes
class EdgeResourceAttribute(Attribute, EdgeMethod, ResourceMethod):
    def __init__(self, name, *args, **kwargs):
        super(EdgeResourceAttribute, self).__init__(name, 'edge', 'resource', *args, **kwargs)

    def check(self, vn, pn, v_link_pair, p_link_pair, method='ge'):
        v, p = vn.edges[v_link_pair], pn.edges[p_link_pair]
        return super()._check_one_element(v, p, method)

    def update_path(self, vl, pn, path, method='+'):
        assert self.type in ['resource']
        assert method in ['+', '-', 'add', 'sub'], NotImplementedError
        assert len(path) > 1
        edges_list = path_to_edges(path)
        for edge in edges_list:
            self.update(vl, pn.edges[edge], method)
        return True


class EdgeExtremaAttribute(Attribute, EdgeMethod, ExtremaMethod):
    def __init__(self, name, *args, **kwargs):
        super(EdgeExtremaAttribute, self).__init__(name, 'edge', 'extrema', *args, **kwargs)

    def update_path(self, vl, pn, path, method='+'):
        return True


class EdgeLatencyAttribute(Attribute, EdgeMethod):
    def __init__(self, name='latency', min=0., max=1., *args, **kwargs):
        super(EdgeLatencyAttribute, self).__init__(name, 'edge', 'fixed', extrema=False, generative=True, distribution='customized', min=min, max=max, *args, **kwargs)

    def generate_data(self, network):
        # convert edge attributes to node attributes
        pos_attr_dict = nx.get_node_attributes(network, 'pos')
        latency_data = []
        for e in network.edges:
            pos_a = np.array(pos_attr_dict[e[0]])
            pos_b = np.array(pos_attr_dict[e[1]])
            latency_data.append(np.linalg.norm(pos_a - pos_b))
        norm_latency_data = np.array(latency_data)
        # norm_latency_data = (norm_latency_data - norm_latency_data.min()) / (norm_latency_data.max() - norm_latency_data.min())
        latency_data = norm_latency_data * (self.max - self.min) + self.min
        return latency_data


ATTRIBUTES_DICT = {
    ('node', 'resource'): NodeResourceAttribute,
    ('node', 'extrema'): NodeExtremaAttribute,
    ('edge', 'resource'): EdgeResourceAttribute,
    ('edge', 'extrema'): EdgeExtremaAttribute,
    ('node', 'position'): NodePositionAttribute,
}


if __name__ == '__main__':
    pass
