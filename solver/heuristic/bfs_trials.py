import networkx as nx
from abc import abstractclassmethod

from ..solver import Solver
from ..rank.node_rank import RWNodeRank, OrderNodeRank, RandomNodeRank


class BFSSolver(Solver):
    
    def __init__(self, controller, recorder, counter, **kwargs):
        super(BFSSolver, self).__init__(controller, recorder, counter, **kwargs)

    def select_action(self, obs):
        return self.solve(obs['v_net'], obs['p_net'])

    @abstractclassmethod
    def solve(self, v_net, p_net):
        raise NotImplementedError


class OrderRankBFSSolver(BFSSolver):

    name = 'order_rank_bfs'

    def __init__(self, controller, recorder, counter, **kwargs):
        super(OrderRankBFSSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.node_rank = OrderNodeRank()

    def solve(self, instance):
        v_net, p_net  = instance['v_net'], instance['p_net']
        v_net_rank = self.node_rank.rank(v_net)
        p_net_rank = self.node_rank.rank(p_net)

        sorted_v_nodes = list(v_net_rank)
        sorted_p_nodes = list(p_net_rank)

        p_net_init_node = sorted_p_nodes[0]
        solution = self.controller.bfs_deploy(v_net, p_net, sorted_v_nodes, p_net_init_node, shortest_method=self.shortest_method)
        return solution


class RandomRankBFSSolver(BFSSolver):

    name = 'random_rank_bfs'

    def __init__(self, controller, recorder, counter, **kwargs):
        super(RandomRankBFSSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.node_rank = RandomNodeRank()

    def solve(self, instance):
        v_net, p_net  = instance['v_net'], instance['p_net']

        v_net_rank = self.node_rank.rank(v_net)
        p_net_rank = self.node_rank.rank(p_net)

        sorted_v_nodes = list(v_net_rank)
        sorted_p_nodes = list(p_net_rank)

        p_net_init_node = sorted_p_nodes[0]
        solution = self.controller.bfs_deploy(v_net, p_net, sorted_v_nodes, p_net_init_node)
        return solution


class RandomWalkRankBFSSolver(BFSSolver):
    
    name = 'rw_rank_bfs'

    def __init__(self, controller, recorder, counter, **kwargs):
        super(RandomWalkRankBFSSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.node_rank = RWNodeRank()

    def solve(self, instance):
        v_net, p_net  = instance['v_net'], instance['p_net']
        
        v_net_nodes_rank = self.node_rank.rank(v_net)
        p_net_nodes_rank = self.node_rank.rank(p_net)
        largest_rank_vid = list(v_net_nodes_rank.keys())[0]
        v_net_node_level = nx.single_source_shortest_path_length(v_net, largest_rank_vid)
        v_net_node_level_rank_list = []
        for node, level in v_net_node_level.items():
            v_net_node_level_rank_list.append({'node': node, 'level': level, 'rank': v_net_nodes_rank[node]})
        sorted_v_net_node = sorted(v_net_node_level_rank_list, key=lambda r: (r['level'], -r['rank']))

        sorted_v_nodes = [n['node'] for n in sorted_v_net_node]
        sorted_p_nodes = [n for n in p_net_nodes_rank]
        p_net_init_node = sorted_p_nodes[0]
        solution = self.controller.bfs_deploy(v_net, p_net, sorted_v_nodes, p_net_init_node)
        return solution
