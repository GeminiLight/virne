# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import copy
import numpy as np
import networkx as nx

from utils import path_to_links, generate_data_with_distribution


"""
To-do
Energy: Node Energy = (pmnode - pbnode)*  (snode.cpu - snode.lastcpu) / snode.cpu + pbnode
"""


def create_attrs_from_setting(attrs_setting):
    attrs = {attr_dict['name']: Attribute.from_dict(attr_dict) for attr_dict in attrs_setting}
    return attrs


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

        assert self.generative in [True, False]

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
                self.min = kwargs.get('min', 0.)
                self.max = kwargs.get('max', 1.)

    @classmethod
    def from_dict(cls, dict):
        dict_copy = copy.deepcopy(dict)
        name = dict_copy.pop('name')
        owner = dict_copy.pop('owner')
        type = dict_copy.pop('type')
        assert (owner, type) in ATTRIBUTES_DICT.keys(), ValueError('Unsupproted attribute!')
        AttributeClass = ATTRIBUTES_DICT.get((owner, type))
        return AttributeClass(name, **dict_copy)

    def get(self, net, id):
        if self.owner == 'node':
            return net.nodes[id][self.name]
        elif self.owner == 'link':
            return net.links[id][self.name]

    def check(self, *args, **kwargs):
        return True

    def size(self, network):
        size = network.num_nodes if self.owner == 'node' else network.num_links
        return size

    def _generate_data_with_dist(self, network):
        assert self.generative
        size = network.num_nodes if self.owner == 'node' else network.num_links
        if self.distribution == 'uniform':
            kwargs = {'low': self.low, 'high': self.high}
        elif self.distribution == 'normal':
            kwargs = {'loc': self.loc, 'scale': self.scale}
        elif self.distribution == 'exponential':
            kwargs = {'scale': self.scale}
        elif self.distribution == 'possion':
            kwargs = {'lam': self.lam}
        elif self.distribution == 'customized':
            data = np.random.uniform(0., 1., size)
            return data * (self.max - self.min) + self.min
        else:
            raise NotImplementedError
        return generate_data_with_distribution(size, distribution=self.distribution, dtype=self.dtype, **kwargs)

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

class LinkInfoAttribute(InfoAttribute):

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, 'link', *args, **kwargs)


### Public Methods ###
class ResourceMethod:

    def update(self, v, p, method='+', safe=True):
        assert self.type in ['resource']
        assert method in ['+', '-', 'add', 'sub']
        if method in ['+', 'add']:
            p[self.name] += v[self.name]
        elif method in ['-', 'sub']:
            if safe:
                assert v[self.name] <= p[self.name], f'{self.name}: (v = {v[self.name]}) > (p = {p[self.name]})'
            p[self.name] -= v[self.name]
        else:
            raise NotImplementedError
        return True
    
    def _check_one_element(self, v, p, method='le'):
        assert method in ['>=', '<=', 'ge', 'le', 'eq']
        if method in ['>=', 'ge']:
            return v[self.name] >= p[self.name], v[self.name] - p[self.name]
        elif method in ['<=', 'le']:
            return v[self.name] <= p[self.name], p[self.name] - v[self.name]
        elif method in ['==', 'eq']:
            return v[self.name] == p[self.name], abs(v[self.name] - p[self.name])
        else:
            raise NotImplementedError(f'Used method {method}')

    def generate_data(self, network):
        if self.generative:  # generative attribute
            return self._generate_data_with_dist(network)
        else:
            raise NotImplementedError


class ExtremaMethod:

    def update(self, v, p, method='+', safe=True):
        return True

    def check(self, v_net, p_net, v_node_id, p_node_id, method='le'):
        return True

    def generate_data(self, network):
        if self.owner == 'node':
            originator_attribute = network.node_attrs[self.originator]
        else:
            originator_attribute = network.link_attrs[self.originator]
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


class LinkMethod:

    def set_data(self, network, attribute_data):
        if not isinstance(attribute_data, dict):
            attribute_data = {e: attribute_data[i] for i, e in enumerate(network.links)}
        nx.set_edge_attributes(network, attribute_data, self.name)

    def get_data(self, network):
        attribute_data = list(nx.get_edge_attributes(network, self.name).values())
        return attribute_data

    def get_adjacency_data(self, network, normalized=False):
        adjacency_data = nx.attr_sparse_matrix(
            network, edge_attr=self.name, normalized=normalized, rc_order=network.nodes).toarray()
        return adjacency_data

    def get_aggregation_data(self, network, aggr='sum', normalized=False):
        assert aggr in ['sum', 'mean', 'max'], NotImplementedError
        attr_sparse_matrix = nx.attr_sparse_matrix(
                network, edge_attr=self.name, normalized=normalized, rc_order=network.nodes).toarray()
        if aggr == 'sum':
            aggregation_data = attr_sparse_matrix.sum(axis=0)
            aggregation_data = np.asarray(aggregation_data)
        elif aggr == 'mean':
            aggregation_data = attr_sparse_matrix.mean(axis=0)
            aggregation_data = np.asarray(aggregation_data)
        elif aggr == 'max':
            aggregation_data = attr_sparse_matrix.max(axis=0)
        return aggregation_data


# Node Attributes
class NodeResourceAttribute(Attribute, NodeMethod, ResourceMethod):
    def __init__(self, name, *args, **kwargs):
        super(NodeResourceAttribute, self).__init__(name, 'node', 'resource', *args, **kwargs)

    def check(self, v_node, p_node, method='le'):
        result, value = super()._check_one_element(v_node, p_node, method)
        return result, value

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
            pos_r = self._generate_data_with_dist(network)
            pos_r = np.clip(pos_r, self.min_r, self.max_r, out=None)
            pos_data = [(x, y, pos_r) for x, y in zip(pos_x, pos_y)]
        elif 'pos' in network.nodes[0].keys():
            pos_data = list(nx.get_node_attributes(network, 'pos').values())
        else:
            return AttributeError('Please specify how to generate data')
        return pos_data

# Link Attributes
class LinkResourceAttribute(Attribute, LinkMethod, ResourceMethod):
    def __init__(self, name, *args, **kwargs):
        super(LinkResourceAttribute, self).__init__(name, 'link', 'resource', *args, **kwargs)

    def check(self, v_link, p_link, method='le'):
        result, value = super()._check_one_element(v_link, p_link, method)
        return result, value

    def update_path(self, vl, p_net, path, method='+', safe=True):
        assert self.type in ['resource']
        assert method in ['+', '-', 'add', 'sub'], NotImplementedError
        assert len(path) > 1
        links_list = path_to_links(path)
        for link in links_list:
            self.update(vl, p_net.links[link], method, safe=safe)
        return True

class LinkExtremaAttribute(Attribute, LinkMethod, ExtremaMethod):
    def __init__(self, name, *args, **kwargs):
        super(LinkExtremaAttribute, self).__init__(name, 'link', 'extrema', *args, **kwargs)

    def update_path(self, vl, p_net, path, method='+', safe=True):
        return True


class LinkLatencyAttribute(Attribute, LinkMethod):
    def __init__(self, name='latency', *args, **kwargs):
        super(LinkLatencyAttribute, self).__init__(name, 'link', 'latency', *args, **kwargs)

    def generate_data(self, network):
        # convert link attributes to node attributes
        if 'pos' in network.nodes[0].keys():
            pos_attr_dict = nx.get_node_attributes(network, 'pos')
            latency_data = []
            for e in network.links:
                pos_a = np.array(pos_attr_dict[e[0]])
                pos_b = np.array(pos_attr_dict[e[1]])
                latency_data.append(np.linalg.norm(pos_a - pos_b))
            norm_latency_data = np.array(latency_data)
            # norm_latency_data = (norm_latency_data - norm_latency_data.min()) / (norm_latency_data.max() - norm_latency_data.min())
            latency_data = norm_latency_data * (self.max - self.min) + self.min
        else:
            latency_data = self._generate_data_with_dist(network)
        return latency_data


class NodePositionQoSAttribute(Attribute, NodeMethod):

    def __init__(self, name='qos_pos', *args, **kwargs):
        super(NodePositionAttribute, self).__init__(name, 'node', 'position', *args, **kwargs)

    def generate_data(self, network):
        if self.generative:
            pos_r = self._generate_data_with_dist(network)
        return pos_r

    def check(self, v_node, p_node, **kwargs):
        pos_p = p_node[self.name]
        pos_v = v_node[self.name]
        distance = ((pos_p[0] - pos_v[0]) ** 2 + (pos_p[1] - pos_v[1]) ** 2) ** 0.5
        if distance < 0:
            pass
        return 


ATTRIBUTES_DICT = {
    # Resource
    ('node', 'resource'): NodeResourceAttribute,
    ('node', 'extrema'): NodeExtremaAttribute,
    ('link', 'resource'): LinkResourceAttribute,
    ('link', 'extrema'): LinkExtremaAttribute,
    # Fixed
    ('node', 'position'): NodePositionAttribute,
    ('link', 'latency'): LinkLatencyAttribute,
    # QoS
    ('node', 'qos_position'): NodePositionQoSAttribute,
    # ('link', 'qos_latency'): LinkLatencyQosAttribute,
}


if __name__ == '__main__':
    pass