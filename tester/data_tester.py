import sys

from .tester import Tester
from data import Network, PhysicalNetwork, VNSimulator, Generator
from data.attribute import *


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
        edge_attrs = [
            {'name': 'bw', 'owner': 'edge', 'type': 'resource', 'generative': True},
            {'name': 'max_bw', 'owner': 'edge', 'type': 'extrema', 'originator': 'bw'}
        ]
        net = Network(node_attrs=node_attrs, edge_attrs=edge_attrs)
        net.generate_topology(num_nodes=num_nodes)
        net.generate_attrs_data()
        net.to_gml('test.gml')
        new_net = Network.from_gml('test.gml')

    def test_physical_network(self):
        pn_setting = self.config.pn_setting
        pn = PhysicalNetwork.from_setting(pn_setting)

    def test_vn_simulator(self):
        vns_setting = self.config.vns_setting
        vn_simulator = VNSimulator.from_setting(vns_setting)
        vn_simulator.renew()
        vn_simulator.renew()

    def test_generator(self, config):
        Generator.generate_dataset(config)
        # print(f'Success: {sys._getframe().f_code.co_name}')
