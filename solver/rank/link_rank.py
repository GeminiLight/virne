from abc import abstractclassmethod
import numpy as np


class LinkRank(object):
    
    def __init__(self, **kwargs):
        super(LinkRank, self).__init__()

    def __call__(self, network, sort=True):        
        return self.rank(network, sort=sort)

    @abstractclassmethod
    def rank(self, network, sort=True):
        pass

    @staticmethod
    def to_dict(link_rank_vector, network, sort=True):
        link_rank_vector = {e: link_rank_vector[i] for i, e in enumerate(network.links)}
        if sort:
            link_rank_vector = sorted(link_rank_vector.items(), reverse=True, key=lambda x: x[1])
            link_rank_vector = {i: v for i, v in link_rank_vector}
        return link_rank_vector


class OrderLinkRank(LinkRank):

    def __init__(self, **kwargs):
        super(OrderLinkRank, self).__init__(**kwargs)

    def rank(self, network, sort=True):
        link_rank_vector = [value for value in range(len(network.links))]
        return self.to_dict(link_rank_vector, network, sort=sort)


class FFDLinkRank(LinkRank):

    def __init__(self, **kwargs):
        super(OrderLinkRank, self).__init__(**kwargs)

    def rank(self, network, sort=True):
        links_data = network.get_link_attrs_data(network.get_attrs('link', 'resource'))
        link_rank_vector = np.array(links_data).sum(axis=0)
        return self.to_dict(link_rank_vector, network, sort=sort)