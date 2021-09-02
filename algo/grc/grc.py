import os
import sys
import copy
from typing import Mapping
import numpy as np
import networkx as nx

file_path_dir = os.path.abspath('.')
if os.path.abspath('.') not in sys.path:
    sys.path.append(file_path_dir)

from config import get_config
from data.service_function_chain import ServiceFunctionChain
from data.physical_network import PhysicalNetwork


class GRC(object):
    r"""A class that implements GRC (Global Resource Control)."""
    def __init__(self, **kwargs):
        self.set_config(kwargs)
        self.sigma = 0.0001
        self.d = 0.95
    
    def set_config(self, config):
        if not isinstance(config, dict):
            config = vars(config)
        self.considered_node_attrs = config.get('node_attrs', ['cpu', 'ram', 'rom'])
        self.considered_edge_attrs = config.get('edge_attrs', ['bw'])
        self.reused_vnf = config.get('reused_vnf', False)
        self.records_dir = config.get('records_dir', 'records')

    def run(self, env):
        for event_id in range(env.num_events):
            env.ready(event_id)
            print(f"\nEvent: id={event_id}, type={env.curr_event['type']}")

            # Leave Event
            if env.curr_event['type'] == 0:
                env.release()
                env.record(show=True)
                env.save_records(self.records_dir, f'/temp_grc_records.csv')
                continue
            
            # Enter Event
            node_result, slots = self.node_mapping(env.curr_sfc, env.pn)
            if node_result:
                link_result, paths = self.link_mapping(env.curr_sfc, env.pn, slots)
                if link_result:
                    # SUCCESS
                    env.step(True, slots, paths)
                    env.record(show=True)
                    env.save_records(self.records_dir, f'/temp_grc_records.csv')
                    continue
            # FAILURE
            env.step(False, {}, {})
            env.record(show=True)
            env.save_records(self.records_dir, f'/temp_grc_records.csv')
            

    def rank(self, network, rtype='array'):
        r"""Caculate the grc vector to rank node

        Args:
            sigma [float]: the pre-set small positive threshold
            d     [float]: weight the nodes attrs and edges attrs
            edge_attr [str]: the attr of edges considered by M

        Returns:
            r [type]: the grc rank vector
        """
        def calc_grc_c(self, network):
            '''Return the data of avalible resource of nodes'''
            free_nodes_data = network.get_nodes_data(self.considered_node_attrs)
            sum_nodes_data = np.array(free_nodes_data).sum(axis=0)
            return sum_nodes_data / sum_nodes_data.sum(axis=0)

        def calc_grc_M(self, network):
            M = nx.attr_sparse_matrix(
                network, edge_attr=self.considered_edge_attrs[0], normalized=True, rc_order=network.nodes).T
            return M

        c = calc_grc_c(self, network)
        M = calc_grc_M(self, network)
        r = c
        delta = np.inf
        while(delta >= self.sigma):
            new_r = (1 - self.d) * c + self.d * M * r
            delta = np.linalg.norm(new_r - r)
            r = new_r
        if rtype == 'dict':
            dict_r = {}
            for i, v in enumerate(r):
                dict_r[i] = v
            return dict_r
        return r

    def node_mapping(self, sfc, pn):
        r"""Attempt to accommodate VNF in appropriate physical node."""
        sfc_rank = self.rank(sfc, rtype='dict')
        pn_rank = self.rank(pn, rtype='dict')
        sfc_grc_sort = sorted(sfc_rank.items(), reverse=True, key=lambda x: x[1])
        pn_grc_sort = sorted(pn_rank.items(), reverse=True, key=lambda x: x[1])
        ordered_sfc_nodes = [v[0] for v in sfc_grc_sort]
        ordered_pn_nodes = [p[0] for p in pn_grc_sort]

        slots = {}
        for vid in ordered_sfc_nodes:
            flag = False
            for pid in ordered_pn_nodes:
                node_reqs = {n_attr: sfc.nodes[vid][n_attr] for n_attr in self.considered_node_attrs}
                if pn.is_candidate_node(pid, attrs=node_reqs):
                    slots[vid] = pid
                    pn.update_node(pid, {k:-v for k, v in node_reqs.items()})
                    if self.reused_vnf == False: ordered_pn_nodes.remove(pid)
                    flag = True
                    break
            # FAILURE
            if flag == False:
                return flag, {}
        # SUCCESS
        return flag, slots

    def link_mapping(self, sfc, pn, slots):
        r"""Seek a path connecting """
        paths = {}
        for e in sfc.edges:
            edge_req = sfc.edges[e][self.considered_edge_attrs[0]]
            # sub_gragh
            free_edges_data = pn.edges.data(self.considered_edge_attrs[0])
            available_egdes = [(u, v) for (u, v, edge_free) in free_edges_data if edge_free >= edge_req]
            temp_graph = nx.Graph()
            temp_graph.add_edges_from(available_egdes)
            try:
                # find shortest path
                path = nx.dijkstra_path(temp_graph, slots[e[0]], slots[e[1]])
                # update link resource
                pn.update_path(path, {self.considered_edge_attrs[0]: -edge_req})
                paths[e] = path
            except:
                # FAILURE
                return False, paths
        # SUCCESS
        return True, paths


if __name__ == '__main__':
    config, _ = get_config()
    grc = GRC(**vars(config))
    pn = PhysicalNetwork(num_nodes=10, id=1)
    sfc = ServiceFunctionChain(num_nodes=10, id=1)
    pn.generate_data(config.node_attrs, config.edge_attrs, 0, 20, 0, 20)
    sfc.generate_data(config.node_attrs, config.edge_attrs, 0, 20, 0, 20)
    print(sfc.edges[(0, 1)])
    grc.rank(sfc)
