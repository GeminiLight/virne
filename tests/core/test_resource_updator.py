from collections import OrderedDict

import pytest

from virne.core.controller.controller import Controller
from virne.core.controller.resource_updator import ResourceUpdator
from virne.network import PhysicalNetwork, VirtualNetwork
from virne.network.attribute import LinkResourceAttribute


def test_multi_resource_node_update_is_atomic_on_validation_failure():
    p_net = PhysicalNetwork()
    p_net.add_node(0, cpu=10.0, gpu=1.0)
    updater = ResourceUpdator(link_resource_attrs=[])

    with pytest.raises(ValueError, match='gpu'):
        updater.update_node_resources(
            p_net,
            0,
            {'cpu': 5.0, 'gpu': 2.0},
            operator='-',
            safe=True,
        )

    assert p_net.nodes[0] == {'cpu': 10.0, 'gpu': 1.0}


def test_multi_resource_link_update_is_atomic_on_validation_failure():
    p_net = PhysicalNetwork()
    p_net.add_edge(0, 1, bw=10.0, burst=1.0)
    updater = ResourceUpdator(link_resource_attrs=[])

    with pytest.raises(ValueError, match='burst'):
        updater.update_link_resources(
            p_net,
            (0, 1),
            {'bw': 5.0, 'burst': 2.0},
            operator='-',
            safe=True,
        )

    assert p_net.links[(0, 1)] == {'bw': 10.0, 'burst': 1.0}


def test_multi_resource_addition_is_atomic_when_an_attribute_is_missing():
    p_net = PhysicalNetwork()
    p_net.add_node(0, cpu=10.0)
    updater = ResourceUpdator(link_resource_attrs=[])

    with pytest.raises(KeyError, match='gpu'):
        updater.update_node_resources(
            p_net,
            0,
            {'cpu': 5.0, 'gpu': 2.0},
            operator='+',
        )

    assert p_net.nodes[0] == {'cpu': 10.0}


def test_path_update_is_atomic_on_validation_failure():
    v_net = VirtualNetwork()
    v_net.add_edge(0, 1, bw=6.0)
    p_net = PhysicalNetwork()
    p_net.add_edge(0, 1, bw=10.0)
    p_net.add_edge(1, 2, bw=5.0)
    updater = ResourceUpdator(link_resource_attrs=[LinkResourceAttribute('bw')])

    with pytest.raises(ValueError, match='bw'):
        updater.update_path_resources(
            v_net,
            p_net,
            (0, 1),
            [0, 1, 2],
            operator='-',
            safe=True,
        )

    assert p_net.links[(0, 1)]['bw'] == 10.0
    assert p_net.links[(1, 2)]['bw'] == 5.0


def test_link_attribute_direct_path_update_is_atomic():
    p_net = PhysicalNetwork()
    p_net.add_edge(0, 1, bw=10.0)
    p_net.add_edge(1, 2, bw=5.0)
    attribute = LinkResourceAttribute('bw')

    with pytest.raises(ValueError, match='bw'):
        attribute.update_path(
            {'bw': 6.0},
            p_net,
            [0, 1, 2],
            method='-',
            safe=True,
        )

    assert p_net.links[(0, 1)]['bw'] == 10.0
    assert p_net.links[(1, 2)]['bw'] == 5.0


def test_controller_deploy_rolls_back_all_prior_updates_on_failure():
    node_attrs_setting = [
        {'name': 'cpu', 'type': 'resource', 'owner': 'node', 'generative': False},
        {'name': 'gpu', 'type': 'resource', 'owner': 'node', 'generative': False},
    ]
    controller = Controller(
        node_attrs_setting=node_attrs_setting,
        link_attrs_setting=[],
        config={
            'reusable': False,
            'matching_mathod': 'greedy',
            'shortest_method': 'k_shortest',
        },
    )
    p_net = PhysicalNetwork()
    p_net.add_node(0, cpu=10.0, gpu=10.0)
    p_net.add_node(1, cpu=1.0, gpu=10.0)
    solution = {
        'result': True,
        'node_slots_info': OrderedDict([
            ((0, 0), {'cpu': 5.0, 'gpu': 1.0}),
            ((1, 1), {'cpu': 2.0, 'gpu': 1.0}),
        ]),
        'link_paths_info': OrderedDict(),
    }

    with pytest.raises(ValueError, match='cpu'):
        controller.deploy(None, p_net, solution)

    assert p_net.nodes[0] == {'cpu': 10.0, 'gpu': 10.0}
    assert p_net.nodes[1] == {'cpu': 1.0, 'gpu': 10.0}
