from scipy import sparse

from virne.network.base_network import BaseNetwork


def test_structural_properties_follow_topology_regeneration():
    """Graph-size properties must reflect the current topology, not a stale cache."""
    network = BaseNetwork()
    network.generate_topology(num_nodes=3, type='path')

    assert network.num_nodes == 3
    assert network.num_links == 2
    assert network.num_edges == 2

    network.generate_topology(num_nodes=5, type='path')

    assert network.num_nodes == network.number_of_nodes() == 5
    assert network.num_links == network.number_of_edges() == 4
    assert network.num_edges == network.number_of_edges() == 4


def test_network_lookup_supports_tuple_node_ids_from_grid_topology():
    network = BaseNetwork()
    network.generate_topology(num_nodes=4, type='grid_2d', m=2, n=2)

    assert (0, 1) in network[(0, 0)]

    network.description = 'grid'
    assert network['description'] == 'grid'


def test_adjacency_matrix_uses_supported_networkx_api_and_preserves_matrix_semantics():
    network = BaseNetwork()
    network.add_edges_from([(0, 1), (1, 2)])

    adjacency = network.adjacency_matrix

    assert isinstance(adjacency, sparse.csr_matrix)
    assert adjacency.shape == (3, 3)
    assert adjacency.toarray().tolist() == [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]
