# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import copy
import numpy as np
import networkx as nx

from virne.utils import path_to_links, generate_data_with_distribution


class BaseAttribute:
    """
    Base Attribute class for network elements (nodes and links)

    Args:
        name (str): the name of the attribute
        owner (str): the owner of the attribute, either 'node' or 'link'
        type (str): the type of the attribute, options: 'resource', 'extrema', 'info', 'position', 'latency'
        *args: additional arguments
        **kwargs: additional keyword arguments
    """

    def __init__(self, name, owner, type, *args, **kwargs):
        self.name = name
        self.owner = owner
        self.type = type
        # for extrema
        if type == 'extrema':
            self.originator = kwargs.get('originator')
        # for generative
        self.generative = bool(kwargs.get('generative', False))

        assert self.generative in [True, False], ValueError('The generative attribute should be boolean')

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

    def get(self, net, id):
        if self.owner == 'node':
            return net.nodes[id][self.name]
        elif self.owner == 'link':
            return net.links[id][self.name]

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
        info = [f'{key}={str(item)}' for key, item in self.__dict__.items()]
        return f"{self.__class__.__name__}({', '.join(info)})"


class NodeAttributeMethod:

    def set_data(self, network, attribute_data):
        if not isinstance(attribute_data, dict):
            attribute_data = {n: attribute_data[i] for i, n in enumerate(network.nodes)}
        nx.set_node_attributes(network, attribute_data, self.name)

    def get_data(self, network):
        attribute_data = list(nx.get_node_attributes(network, self.name).values())
        return attribute_data


class LinkAttributeMethod:

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
        assert aggr in ['sum', 'mean', 'max', 'min'], NotImplementedError
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
        elif aggr == 'min':
            aggregation_data = attr_sparse_matrix.min(axis=0)
        return aggregation_data


class GraphAttributeMethod:

    def set_data(self, network, attribute_data):
        network.graph[self.name] = attribute_data

    def get_data(self, network):
        attribute_data = network.graph[self.name]
        return attribute_data


class InformationAttribute(BaseAttribute):

    def __init__(self, name, owner, type, *args, **kwargs):
        super().__init__(name, owner, type, *args, **kwargs)
        self.is_constraint = False

class ConstraintAttribute(BaseAttribute):
    """
    """
    def __init__(self, name, owner, type, *args, **kwargs):
        super().__init__(name, owner, type, *args, **kwargs)
        self.is_constraint = True
        self.constraint_restrictions = kwargs.get('restriction', 'hard')
        assert self.constraint_restrictions in ['hard', 'soft']

    def check_constraint_satisfiability(self, v, p, method='le'):
        raise NotImplementedError(f'The attribute {self.name} has not implemented the check_constraint_satisfiability method')

    def _calculate_satisfiability_values(self, v_value, p_value, method='le'):
        assert method in ['>=', '<=', 'ge', 'le', 'eq']
        """
        Calculate the difference between the value of the attribute in the virtual network and the value of the attribute in the physical network

        Offset = Requirement - Availability
        Violation = Max(0, Offset)
        
        Args:
            v_value (float): the value of the attribute in the virtual network
            p_value (float): the value of the attribute in the physical network
            method (str): the method to compare the values, options: '>=', '<=', '=='

        Returns:
            flag (bool): the comparison result
            diff (float): the difference between the values
        """
        if method in ['>=', 'ge']:
            flag, offset = v_value >= p_value, p_value - v_value
            violation = max(0, p_value - v_value)
        elif method in ['<=', 'le']:
            flag, offset = v_value <= p_value, v_value - p_value
            violation = max(0, v_value - p_value)
        elif method in ['==', 'eq']:
            flag, offset = v_value == p_value, abs(v_value - p_value)
            violation = max(0, abs(v_value - p_value))
        else:
            raise NotImplementedError(f'Used method {method}')
        if self.constraint_restrictions == 'hard':
            return flag, offset
        elif self.constraint_restrictions == 'soft':
            return True, offset

class ResourceAttributeMethod:

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

    def generate_data(self, network):
        if self.generative:  # generative attribute
            return self._generate_data_with_dist(network)
        else:
            raise NotImplementedError


class ExtremaAttributeMethod:

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
