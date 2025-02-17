import numpy as np
import networkx as nx

from virne.utils import path_to_links
from .base_attribute import InformationAttribute, ConstraintAttribute, GraphAttributeMethod, ResourceAttributeMethod, ExtremaAttributeMethod


class GraphStatusAttribute(InformationAttribute, GraphAttributeMethod):

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, 'graph', 'status', *args, **kwargs)


class GraphExtremaAttribute(InformationAttribute, GraphAttributeMethod, ExtremaAttributeMethod):

    def __init__(self, name, *args, **kwargs):
        super(GraphExtremaAttribute, self).__init__(name, 'graph', 'extrema', *args, **kwargs)


class GraphResourceAttribute(ConstraintAttribute, GraphAttributeMethod, ResourceAttributeMethod):

    def __init__(self, name, *args, **kwargs):
        super(GraphResourceAttribute, self).__init__(name, 'graph', 'resource', *args, **kwargs)
        self.constraint_restrictions = kwargs.get('constraint_restrictions', 'hard')
        self.checking_level = kwargs.get('checking_level', 'graph')

    def check_constraint_satisfiability(self, v_net, p_net, method='le'):
        v_value, p_value = v_net.graph[self.name], p_net.graph[self.name]
        return self._calculate_satisfiability_values(v_value, p_value, method)

    def update(self, v_net, p_net, method='+', safe=True):
        assert self.type in ['resource']
        assert method in ['+', '-', 'add', 'sub'], NotImplementedError
        if method in ['+', 'add']:
            v_net.graph[self.name] += p_net.graph[self.name]
        elif method in ['-', 'sub']:
            v_net.graph[self.name] -= p_net.graph[self.name]
        return True
