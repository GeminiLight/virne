import numpy as np
import networkx as nx

from virne.utils import path_to_links
from .base_attribute import InformationAttribute, ConstraintAttribute, LinkAttributeMethod, ResourceAttributeMethod, ExtremaAttributeMethod


class LinkStatusAttribute(InformationAttribute, LinkAttributeMethod):

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, 'link', 'status', *args, **kwargs)


class LinkExtremaAttribute(InformationAttribute, LinkAttributeMethod, ExtremaAttributeMethod):
    
    def __init__(self, name, *args, **kwargs):
        super(LinkExtremaAttribute, self).__init__(name, 'link', 'extrema', *args, **kwargs)


class LinkResourceAttribute(ConstraintAttribute, LinkAttributeMethod, ResourceAttributeMethod):

    def __init__(self, name, *args, **kwargs):
        super(LinkResourceAttribute, self).__init__(name, 'link', 'resource', *args, **kwargs)
        self.constraint_restrictions = kwargs.get('constraint_restrictions', 'hard')
        self.checking_level = kwargs.get('checking_level', 'link')

    def check_constraint_satisfiability(self, v, p, method='le'):
        v_value, p_value = v[self.name], p[self.name]
        return self._calculate_satisfiability_values(v_value, p_value, method)

    def update_path(self, vl, p_net, path, method='+', safe=True):
        assert self.type in ['resource']
        assert method in ['+', '-', 'add', 'sub'], NotImplementedError
        assert len(path) > 1
        links_list = path_to_links(path)
        for link in links_list:
            self.update(vl, p_net.links[link], method, safe=safe)
        return True


class LinkLatencyAttribute(ConstraintAttribute, LinkAttributeMethod):

    def __init__(self, name='latency', *args, **kwargs):
        super(LinkLatencyAttribute, self).__init__(name, 'link', 'latency', *args, **kwargs)
        self.constraint_restrictions = kwargs.get('constraint_restrictions', 'hard')
        self.checking_level = kwargs.get('checking_level', 'path')

    def check_constraint_satisfiability(self, v_link, p_path, method='ge'):
        assert method in ['>=', '<=', 'ge', 'le', 'eq']
        p_cum_value = sum([p_link[self.name] for p_link in p_path])
        v_value = v_link[self.name]
        return self._calculate_satisfiability_values(v_value, p_cum_value, method)

    def generate_data(self, network):
        if self.generative and self.distribution == 'position':
            return self._generate_data_with_position(network)
        elif self.generative:
            return self._generate_data_with_dist(network)
        else:
            return NotImplementedError

    def _generate_data_with_position(self, network):
        pos_node_attrs = nx.get_node_attrs(types='position')
        assert len(pos_node_attrs) > 0, AttributeError('The generation of this attribute requires node position')
        pos_node_attr_name = list(pos_node_attrs.keys())[0]
        pos_attr_dict = nx.get_node_attributes(network, pos_node_attr_name)
        latency_data = []
        for e in network.links:
            pos_a = np.array(pos_attr_dict[e[0]])
            pos_b = np.array(pos_attr_dict[e[1]])
            latency_data.append(np.linalg.norm(pos_a - pos_b))
        norm_latency_data = np.array(latency_data)
        latency_data = norm_latency_data * (self.max - self.min) + self.min
        return latency_data
    