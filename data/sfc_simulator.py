import ast
import numpy as np
import pandas as pd

import os
import sys
file_path_dir = os.path.abspath('.')
if os.path.abspath('.') not in sys.path:
    sys.path.append(file_path_dir)

from config import get_config

from data.service_function_chain import ServiceFunctionChain

class SfcSimulator(object):
    def __init__(self, **kwargs):
        """A simulator of SFC"""
        super(SfcSimulator, self).__init__()
        self.set_config(kwargs)
        self.sfcs = []
        self.events = []
        
    def set_config(self, config):
        if not isinstance(config, dict):
            config = vars(config)
        # the number of sfcs
        self.num_sfcs = config.get('num_sfcs', 2000)
        # the node and edge attributes of sfcs
        self.node_attrs = config.get('node_attrs', ['cpu', 'ram', 'rom'])
        self.edge_attrs = config.get('edge_attrs', ['bw'])
        # the minximum and maximum of sfcs' length
        self.min_length = config.get('min_length', 2)
        self.max_length = config.get('max_length', 15)
        # the minximum and maximum of the node attributes' value of sfcs
        self.min_node_request = config.get('min_node_request', 2)
        self.max_node_request = config.get('max_node_request', 30)
        # the minximum and maximum of the edge attributes' value of sfcs
        self.min_edge_request = config.get('min_edge_request', 2)
        self.max_edge_request = config.get('max_edge_request', 30)
        # the arverge arrival rate of sfcs
        self.aver_arrival_rate = config.get('aver_arrival_rate', 20)
        # the arverge lifetime of sfcs
        self.aver_lifetime = config.get('aver_lifetime', 500)

    def renew(self, sfcs=True, events=True):
        if sfcs == True:
            self.renew_sfcs()
        if events == True:
            self.renew_events()
        return self.sfcs, self.events

    def renew_sfcs(self):
        self.arrange_sfcs()
        for i in range(self.num_sfcs):
            sfc = ServiceFunctionChain(num_nodes=self.sfcs_length[i], id=i,
                                        arrival_time=self.sfcs_arrival_time[i],
                                        lifetime=self.sfcs_lifetime[i])
            sfc.generate_data(self.node_attrs, self.edge_attrs,
                                self.min_node_request, self.max_node_request,
                                self.min_edge_request, self.max_edge_request)
            self.sfcs.append(sfc)
        return self.sfcs

    def renew_events(self):
        arrival_list = [{'sfc_id': sfc.id, 'time': sfc.arrival_time, 'type': 1} for sfc in self.sfcs]
        leave_list = [{'sfc_id': sfc.id, 'time': sfc.arrival_time+sfc.lifetime, 'type': 0} for sfc in self.sfcs]
        event_list = arrival_list + leave_list
        self.events = sorted(event_list, key=lambda e: e.__getitem__('time'))
        for i, e in enumerate(self.events): e['id'] = i
        return self.events

    def arrange_sfcs(self):
        self.sfcs_length = np.random.randint(self.min_length, self.max_length, self.num_sfcs)
        self.sfcs_lifetime = np.random.exponential(self.aver_lifetime, self.num_sfcs)
        self.sfcs_arrival_time = np.random.poisson(50, self.num_sfcs)
        for i in range(self.num_sfcs):
            self.sfcs_arrival_time[i] += 100 * int(i/self.aver_arrival_rate)
            
    def to_csv(self, sfc_path, event_path):
        r"""
        Save the sfcs data and events data of physical network to `sfc_path`, `event_path`.

        Args:
            sfc_path (list): List of sfcs object.
            event_path ([type]): List of events information.
        """
        sfc_list = []
        for sfc in self.sfcs:
            nodes_data = {n_attr: sfc.get_node_attr(n_attr) for n_attr in self.node_attrs}
            edges_data = {e_attr: sfc.get_edge_attr(e_attr) for e_attr in self.edge_attrs}
            sfc_info = {
                'id': sfc.id,
                'num_nodes': sfc.num_nodes,
                'edge_list': sfc.edges,
                'lifetime': sfc.lifetime,
                'arrival_time': sfc.arrival_time,
                'nodes_data': nodes_data,
                'edges_data': edges_data,
            }
            sfc_list.append(sfc_info)
        pd_sfcs = pd.DataFrame.from_dict(sfc_list)
        pd_sfcs[['id', 'num_nodes']] = pd_sfcs[['id', 'num_nodes']].astype('int')
        pd_sfcs.to_csv(sfc_path)

        pf_events = pd.DataFrame(self.events)
        pf_events[['id', 'sfc_id', 'type']] = pf_events[['id', 'sfc_id', 'type']].astype('int')
        pf_events.to_csv(event_path)

    @staticmethod
    def from_csv(sfc_path, event_path):
        sfcs_data = pd.read_csv(sfc_path, index_col=[0])
        events_data = pd.read_csv(event_path, index_col=[0])
        sfc_simulator = SfcSimulator(num_sfcs=len(sfcs_data))
        for id, sfc_info in sfcs_data.iterrows():
            # ast.literal_eval(sfc_info)
            sfc = ServiceFunctionChain(num_nodes=sfc_info['num_nodes'],
                                        id=sfc_info['id'],
                                        arrival_time=sfc_info['arrival_time'],
                                        lifetime=sfc_info['lifetime'])
            for attr_name, attr_data in ast.literal_eval(sfc_info['nodes_data']).items():
                sfc.set_node_attr(attr_name, attr_data)
            for attr_name, attr_data in ast.literal_eval(sfc_info['edges_data']).items():
                sfc.set_edge_attr(attr_name, attr_data)
            sfc_simulator.sfcs.append(sfc)
        sfc_simulator.events = [event.to_dict() for id, event in events_data.iterrows()]
        return sfc_simulator


if __name__ == '__main__':
    config, _ = get_config()
    # generate sfcs dataset
    sfc_simulator = SfcSimulator(type='actor', **vars(config))
    sfc_simulator.renew_sfcs()
    sfc_simulator.renew_events()
    sfc_simulator.to_csv(config.sfcs_dataset, config.events_dataset)
    new_sfc_simulator = SfcSimulator.from_csv(config.sfcs_dataset, config.events_dataset)
