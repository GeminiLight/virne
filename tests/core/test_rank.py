import math

import pytest

from virne.network import PhysicalNetwork, VirtualNetwork
from virne.solver.rank.link_rank import OrderLinkRank
from virne.solver.rank.node_rank import GRCNodeRank, NPSNodeRank, RWNodeRank, RandomNodeRank


HARD_NODE_ATTRS = [
    {'name': 'cpu', 'type': 'resource', 'owner': 'node', 'generative': False},
]
HARD_LINK_ATTRS = [
    {'name': 'bw', 'type': 'resource', 'owner': 'link', 'generative': False},
]


def make_p_net(nodes, edges):
    network = PhysicalNetwork(
        config={
            'node_attrs_setting': HARD_NODE_ATTRS,
            'link_attrs_setting': HARD_LINK_ATTRS,
        }
    )
    network.add_nodes_from((node_id, {'cpu': float(cpu)}) for node_id, cpu in nodes)
    network.add_edges_from((u, v, {'bw': float(bw)}) for u, v, bw in edges)
    return network


def make_v_net(nodes, edges):
    network = VirtualNetwork(
        config={
            'node_attrs_setting': HARD_NODE_ATTRS,
            'link_attrs_setting': HARD_LINK_ATTRS,
        },
        id=1,
        arrival_time=0.0,
        lifetime=1.0,
    )
    network.add_nodes_from((node_id, {'cpu': float(cpu)}) for node_id, cpu in nodes)
    network.add_edges_from((u, v, {'bw': float(bw)}) for u, v, bw in edges)
    return network


def make_grid_network():
    network = PhysicalNetwork(
        config={
            'node_attrs_setting': HARD_NODE_ATTRS,
            'link_attrs_setting': HARD_LINK_ATTRS,
        }
    )
    network.generate_topology(num_nodes=4, type='grid_2d', m=2, n=2)
    for node_id in network.nodes:
        network.nodes[node_id]['cpu'] = 10.0
    for link in network.links:
        network.links[link]['bw'] = 10.0
    return network


def test_random_and_nps_ranks_do_not_treat_node_ids_as_positions():
    grid = make_grid_network()
    random_rank = RandomNodeRank().rank(grid)
    assert set(random_rank) == set(grid.nodes)
    assert all(math.isfinite(value) for value in random_rank.values())

    v_net = make_v_net([(10, 1), (20, 1)], [(10, 20, 1)])
    nps_rank = NPSNodeRank().rank(v_net)
    assert set(nps_rank) == {10, 20}


def test_order_link_rank_preserves_link_insertion_order():
    network = make_p_net([(0, 1), (1, 1), (2, 1)], [(0, 1, 1), (1, 2, 1)])

    assert list(OrderLinkRank().rank(network)) == list(network.links)


def test_grc_zero_resources_returns_finite_ranks():
    network = make_p_net([(0, 0), (1, 0)], [(0, 1, 0)])

    rank = GRCNodeRank().rank(network)

    assert all(math.isfinite(value) for value in rank.values())


@pytest.mark.parametrize(
    'factory, message',
    [
        (lambda: GRCNodeRank(sigma=0), 'sigma'),
        (lambda: GRCNodeRank(d=1), 'd'),
        (lambda: GRCNodeRank(max_iterations=0), 'max_iterations'),
        (lambda: RWNodeRank(sigma=0), 'sigma'),
        (lambda: RWNodeRank(p_J_u=0.8, p_F_u=0.8), 'sum'),
        (lambda: RWNodeRank(max_iterations=0), 'max_iterations'),
    ],
)
def test_iterative_ranks_reject_invalid_parameters(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_random_walk_reports_non_convergence_instead_of_hanging():
    network = make_p_net([(0, 1), (1, 2)], [(0, 1, 1)])
    ranker = RWNodeRank(p_J_u=0, p_F_u=1, max_iterations=3)

    with pytest.raises(RuntimeError, match='did not converge'):
        ranker.rank(network)
