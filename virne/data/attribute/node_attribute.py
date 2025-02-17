import numpy as np
import networkx as nx

from .base_attribute import InformationAttribute, ConstraintAttribute, NodeAttributeMethod, ResourceAttributeMethod, ExtremaAttributeMethod


class NodeStatusAttribute(InformationAttribute, NodeAttributeMethod):

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, 'node', 'status', *args, **kwargs)


class NodeExtremaAttribute(InformationAttribute, NodeAttributeMethod, ExtremaAttributeMethod):

    def __init__(self, name, *args, **kwargs):
        super(NodeExtremaAttribute, self).__init__(name, 'node', 'extrema', *args, **kwargs)


class NodeResourceAttribute(ConstraintAttribute, NodeAttributeMethod, ResourceAttributeMethod):

    def __init__(self, name, *args, **kwargs):
        super(NodeResourceAttribute, self).__init__(name, 'node', 'resource', *args, **kwargs)
        self.constraint_restrictions = kwargs.get('constraint_restrictions', 'hard')
        self.checking_level = kwargs.get('checking_level', 'node')

    def check_constraint_satisfiability(self, v, p, method='le'):
        v_value, p_value = v[self.name], p[self.name]
        return self._calculate_satisfiability_values(v_value, p_value, method)

# class NodePositionQoSAttribute(InformationAttribute, NodeAttributeMethod):

#     def __init__(self, name='qos_pos', *args, **kwargs):
#         super(NodePositionQoSAttribute, self).__init__(name, 'node', 'position', *args, **kwargs)

#     def generate_data(self, network):
#         if self.generative:
#             pos_r = self._generate_data_with_dist(network)
#         return pos_r

class NodePositionAttribute(ConstraintAttribute, NodeAttributeMethod):

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

    # def check(self, v_node, p_node, **kwargs):
    #     pos_p = p_node[self.name]
    #     pos_v = v_node[self.name]
    #     distance = ((pos_p[0] - pos_v[0]) ** 2 + (pos_p[1] - pos_v[1]) ** 2) ** 0.5
    #     if distance < 0:
    #         pass
    #     return 


"""
To-do
Energy: Node Energy = (pmnode - pbnode)*  (snode.cpu - snode.lastcpu) / snode.cpu + pbnode
"""
