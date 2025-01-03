import numpy as np
import networkx as nx

from virne.utils import path_to_links
from .base_attribute import Attribute, InfoAttribute, LinkAttributeMethod, ResourceAttributeMethod, ExtremaAttributeMethod


class LinkInfoAttribute(InfoAttribute, LinkAttributeMethod):

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, 'link', *args, **kwargs)


class LinkResourceAttribute(Attribute, LinkAttributeMethod, ResourceAttributeMethod):
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


class LinkExtremaAttribute(Attribute, LinkAttributeMethod, ExtremaAttributeMethod):
    def __init__(self, name, *args, **kwargs):
        super(LinkExtremaAttribute, self).__init__(name, 'link', 'extrema', *args, **kwargs)

    def update_path(self, vl, p_net, path, method='+', safe=True):
        return True


class LinkLatencyAttribute(Attribute, LinkAttributeMethod):

    def __init__(self, name='latency', *args, **kwargs):
        super(LinkLatencyAttribute, self).__init__(name, 'link', 'latency', *args, **kwargs)
        if self.generative and self.distribution == 'position':
            self.max = kwargs.get('max', 1.)  # the maximum value of the latency
            self.min = kwargs.get('min', 0.)  # the minimum value of the latency

    def check(self, v_link, p_path, method='le'):
        assert method in ['>=', '<=', 'ge', 'le', 'eq']
        p_cum_value = [p_link[self.name] for p_link in p_path]
        v_value = v_link[self.name]
        return self._get_check_results(v_value, p_cum_value, method)

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
    