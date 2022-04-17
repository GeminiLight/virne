import networkx as nx
from abc import abstractclassmethod

from base.controller import Controller
from ..solver import Solver
from ..rank.node_rank import RWNodeRank, OrderNodeRank, RandomNodeRank


class BFSSolver(Solver):
    
    def __init__(self, name, reusable=False, verbose=1, **kwargs):
        super(BFSSolver, self).__init__(name, reusable=reusable, verbose=verbose, **kwargs)

    def select_action(self, obs):
        return self.solve(obs['vn'], obs['pn'])

    @abstractclassmethod
    def solve(self, vn, pn):
        raise NotImplementedError


class OrderRankBFSSolver(BFSSolver):

    def __init__(self, reusable=False, verbose=1, **kwargs):
        super(OrderRankBFSSolver, self).__init__('order_rank_bfs', reusable=reusable, verbose=verbose, **kwargs)
        self.node_rank = OrderNodeRank()

    def solve(self, instance):
        vn, pn  = instance['vn'], instance['pn']
        vn_rank = self.node_rank.rank(vn)
        pn_rank = self.node_rank.rank(pn)

        sorted_v_nodes = list(vn_rank)
        sorted_p_nodes = list(pn_rank)

        pn_init_node = sorted_p_nodes[0]
        solution = Controller.bfs_deploy(vn, pn, sorted_v_nodes, pn_init_node)
        return solution


class RandomRankBFSSolver(BFSSolver):

    def __init__(self, reusable=False, verbose=1, **kwargs):
        super(RandomRankBFSSolver, self).__init__('random_rank_bfs', reusable=reusable, verbose=verbose, **kwargs)
        self.node_rank = RandomNodeRank()

    def solve(self, instance):
        vn, pn  = instance['vn'], instance['pn']

        vn_rank = self.node_rank.rank(vn)
        pn_rank = self.node_rank.rank(pn)

        sorted_v_nodes = list(vn_rank)
        sorted_p_nodes = list(pn_rank)

        pn_init_node = sorted_p_nodes[0]
        solution = Controller.bfs_deploy(vn, pn, sorted_v_nodes, pn_init_node)
        return solution


class RandomWalkRankBFSSolver(BFSSolver):
    
    def __init__(self, reusable=False, verbose=1, **kwargs):
        super(RandomWalkRankBFSSolver, self).__init__('rw_rank_bfs', reusable=reusable, verbose=verbose, **kwargs)
        self.node_rank = RWNodeRank()

    def solve(self, instance):
        vn, pn  = instance['vn'], instance['pn']
        
        vn_nodes_rank = self.node_rank.rank(vn)
        pn_nodes_rank = self.node_rank.rank(pn)
        largest_rank_vid = list(vn_nodes_rank.keys())[0]
        vn_node_level = nx.single_source_shortest_path_length(vn, largest_rank_vid)
        vn_node_level_rank_list = []
        for node, level in vn_node_level.items():
            vn_node_level_rank_list.append({'node': node, 'level': level, 'rank': vn_nodes_rank[node]})
        sorted_vn_node = sorted(vn_node_level_rank_list, key=lambda r: (r['level'], -r['rank']))

        sorted_v_nodes = [n['node'] for n in sorted_vn_node]
        sorted_p_nodes = [n for n in pn_nodes_rank]
        pn_init_node = sorted_p_nodes[0]
        solution = Controller.bfs_deploy(vn, pn, sorted_v_nodes, pn_init_node)
        return solution
