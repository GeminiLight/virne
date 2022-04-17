from abc import abstractclassmethod
import numpy as np


class EdgeRank(object):
    
    def __init__(self, **kwargs):
        super(EdgeRank, self).__init__()

    def __call__(self, network, sort=True):        
        return self.rank(network, sort=sort)

    @abstractclassmethod
    def rank(self, network, sort=True):
        pass

    @staticmethod
    def to_dict(edge_rank_vector, network, sort=True):
        edge_rank_vector = {e: edge_rank_vector[i] for i, e in enumerate(network.edges)}
        if sort:
            edge_rank_vector = sorted(edge_rank_vector.items(), reverse=True, key=lambda x: x[1])
            edge_rank_vector = {i: v for i, v in edge_rank_vector}
        return edge_rank_vector



class OrderEdgeRank(EdgeRank):

    def __init__(self, **kwargs):
        super(OrderEdgeRank, self).__init__(**kwargs)

    def rank(self, network, sort=True):
        edge_rank_vector = [value for value in range(len(network.edges))]
        return self.to_dict(edge_rank_vector, network, sort=sort)


class FFDEdgeRank(EdgeRank):

    def __init__(self, **kwargs):
        super(OrderEdgeRank, self).__init__(**kwargs)

    def rank(self, network, sort=True):
        edges_data = network.get_edge_data(network.get_attrs('edge', 'resource'))
        edge_rank_vector = np.array(edges_data).sum(axis=0)
        return self.to_dict(edge_rank_vector, network, sort=sort)