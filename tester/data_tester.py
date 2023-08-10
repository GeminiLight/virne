import sys

from .tester import Tester
from virne.data import Network, PhysicalNetwork, VirtualNetworkRequestSimulator, Generator
from virne.data.attribute import *


class DataTester(Tester):

    def __init__(self):
        super(DataTester, self).__init__()

    def test_attribute(self):
        pass

    def test_network(self):
        num_nodes = 10
        node_attrs = [
            {'name': 'cpu', 'owner': 'node', 'type': 'resource', 'generative': True},
            {'name': 'max_cpu', 'owner': 'node', 'type': 'extrema', 'originator': 'cpu'}
        ]
        link_attrs = [
            {'name': 'bw', 'owner': 'edge', 'type': 'resource', 'generative': True},
            {'name': 'max_bw', 'owner': 'edge', 'type': 'extrema', 'originator': 'bw'}
        ]
        net = Network(node_attrs=node_attrs, link_attrs=link_attrs)
        net.generate_topology(num_nodes=num_nodes)
        net.generate_attrs_data()
        net.to_gml('test.gml')
        new_net = Network.from_gml('test.gml')

    def test_physical_network(self):
        p_net_setting = self.config.p_net_setting
        p_net = PhysicalNetwork.from_setting(p_net_setting)

    def test_v_net_simulator(self):
        v_sim_setting = self.config.v_sim_setting
        v_net_simulator = VirtualNetworkRequestSimulator.from_setting(v_sim_setting)
        v_net_simulator.renew()
        v_net_simulator.renew()

    def test_generator(self, config):
        Generator.generate_dataset(config)
        # print(f'Success: {sys._getframe().f_code.co_name}')
