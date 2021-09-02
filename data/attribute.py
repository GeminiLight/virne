import numpy as np

class Attribute:
    def __init__(self, name, type='node', low=None, high=None, dist='uniform', dtype='int', **kwargs):
        r"""Base class for all attributes.

        Args:
            name (str): Attribute name
            type (str): Specify whether it is an node or egde attribute.
            low (int or float, optional): Low bound of attribute values for generating data
            high (int or float, optional): High bound of attribute values for generating data
            dtype (str): Type of attribute values
        """
        self.name = name
        self.type = type
        self.low = low
        self.high = high
        self.dtype = dtype

        for k, v in kwargs.items():
            self[k] = v

    def assign(self, network, attr_data):
        r"""Assign the attribute values of nework with `attr_data`.

        Args:
            attr_data (list or dict): Attribute data
        """
        pass

    def data(self, network):
        r"""Return the attribute data of network."""
        return network.get_node_attr(self.name, rtype='list')

    def number(self, network):
        r"""Return the number of attribute value."""
        try:
            return len(self.data(network))
        finally:
            if self.type == 'node':
                return network.num_nodes
            elif self.type == 'edge':
                return network.num_edges

    def generate(self, network):
        if self.dtype not in ['int', 'float']:
            raise ValueError(f'Attribute with type {self.dtype} don\'t support such operation.')
        size = self.number(network)
        if self.dist == 'normal':
            data = np.random.normal(self.low, self.high, size, self.dtype)
        elif self.dtype == 'int':
            data = np.random.randint(self.low, self.high, size)
        elif self.dtype == 'float':
            data = np.random.uniform(self.low, self.high, size)
        self.assign(network, data)

    def compare(self):
        pass

    def update(self):
        pass

    def max_value(self, network):
        if self.dtype != 'int':
            raise ValueError(f'Attribute with type {self.dtype} don\'t support such operation.')
        data = self.data(network)
        return max(data)

    def min_value(self, network):
        if self.dtype != 'int':
            raise ValueError(f'Attribute with type {self.dtype} don\'t support such operation.')
        data = self.data(network)
        return min(data)

    def __setitem__(self, key: str, value):
        r"""Sets the attribute key to value."""
        setattr(self, key, value)

class CompareAttribute(Attribute):
    def __init__(self):
        pass

class ConsumeAttribute(Attribute):
    pass

if __name__ == '__main__':
    a = Attribute('cpu', abc=123)
    