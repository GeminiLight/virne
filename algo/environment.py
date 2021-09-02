import os
import sys
import copy
import numpy as np
import pandas as pd
from abc import abstractmethod, ABCMeta

file_path_dir = os.path.abspath('.')
if file_path_dir not in sys.path:
    sys.path.append(file_path_dir)

from config import get_config
from data import PhysicalNetwork, SfcSimulator


class Environment(metaclass=ABCMeta):
    r"""A general environment for various algorithms based on heuristics and RL"""
    def __init__(self, **kwargs):
        super(Environment, self).__init__()
        self.set_config(kwargs)
        self.pn = PhysicalNetwork()
        self.sfc_simulator = SfcSimulator(**kwargs)
        self.num_events = self.sfc_simulator.num_sfcs * 2
        self.reset()

    def set_config(self, config):
        r"""Set the configuration of environment."""
        if not isinstance(config, dict):
            config = vars(config)
        self.pn_node_dataset = config.get('pn_node_dataset', 'dataset/pn/nodes_data.csv')
        self.pn_edge_dataset = config.get('pn_edge_dataset', 'dataset/pn/edges_csv.csv')
        self.sfcs_dataset = config.get('sfcs_dataset', 'dataset/sfc/sfcs_data.csv')
        self.events_dataset = config.get('events_dataset', 'dataset/sfc/events_data.csv')
        self.num_sfcs = config.get('num_sfcs', 2000)
        self.node_attrs = config.get('node_attrs', ['cpu', 'ram', 'rom'])
        self.edge_attrs = config.get('edge_attrs', ['bw'])
        self.records_dir = config.get('records_dir')

    def reset(self):
        r"""Reset the environment."""
        self.pn = PhysicalNetwork.from_csv(self.pn_node_dataset, self.pn_edge_dataset)
        # self.sfc_simulator.renew(sfcs=True, events=True)
        self.sfc_simulator = SfcSimulator.from_csv(self.sfcs_dataset, self.events_dataset)
        self.pn_backup = copy.deepcopy(self.pn)
        self.curr_event_id = 0
        self.success_flag = False
        # records
        self.total_revenue = 0
        self.total_cost = 0
        self.success = 0
        self.inservice = 0
        self.records = [] # {'event_id': int, 'event_type': bool, 'sfc_id': int, 'result': bool, 'slots': dict, 'paths': dict}

    def ready(self, event_id=0):
        r"""Ready to attempt to execuate the current events."""
        self.pn_backup = copy.deepcopy(self.pn)

        self.curr_event_id = int(event_id)
        self.curr_event = self.sfc_simulator.events[event_id]

        self.curr_sfc_id = int(self.curr_event['sfc_id'])
        self.curr_sfc = self.sfc_simulator.sfcs[self.curr_sfc_id]
        self.curr_sfc_revenue = self.curr_sfc.sum_requests
        self.curr_sfc_cost = 0

        self.success_flag = False

        self.curr_vnf_id = 0
        self.update_flag = True
        self.selected_nodes = []
        self.lastest_action = -1

        self.node_slots = {}
        self.edge_paths = {}
        
    @abstractmethod
    def step(self):
        """Environment receives the agent's decision and returns corresponding feedback.
        
        An abstract method which must be implemented to adapt to your algorithm

        An Exmple:

        IF SUCCESS:
            # inserver & success
            self.inservice += 1
            self.success += 1

            # slots & paths
            self.node_slots
            self.edge_paths

            # cost & revenue
            self.curr_sfc_cost = self.cal_curr_sfc_cost()
            self.total_revenue += self.curr_sfc_revenue
            self.total_cost += self.curr_sfc_cost
            return True

        IF FAILURE:
            self.curr_sfc_revenue = 0
            self.pn = copy.deepcopy(self.pn_backup)
            return False
        """
        pass

    def release(self):
        r"""Release occupied resources when a SFC leaves PN."""
        sfc_event_pair = {record['sfc_id']: record['event_id'] for record in self.records}
        record_id = sfc_event_pair[self.curr_sfc_id]
        record = self.records[record_id]
        if record['result'] == False:
            pass
        else:
            for vnf_id, pid in record['slots'].items():
                vnf_requests_dict = self.curr_sfc.nodes[vnf_id]
                self.pn.update_node(pid, vnf_requests_dict)
            for vl_pair, path in record['paths'].items():
                vl_requests_dict = self.curr_sfc.edges[vl_pair]
                self.pn.update_path(path, vl_requests_dict)
            self.pn_backup = copy.deepcopy(self.pn)
            self.inservice -= 1
        return True
    
    def calculate_curr_sfc_cost(self):
        r"""Calculate the deployment cost of current sfc according to `edge paths`"""
        curr_sfc_cost = 0
        for edge, path in self.edge_paths.items():
            revenue = self.curr_sfc.edges[edge][self.edge_attrs[0]]
            curr_sfc_cost += revenue * (len(path) - 2)
        curr_sfc_cost += self.curr_sfc_revenue
        return curr_sfc_cost

    def record(self, show=False):
        r"""After executing an event, the result is recorded."""
        if self.curr_event['type'] and self.success_flag:
            revenue = self.curr_sfc_revenue
            cost = self.curr_sfc_cost
        else:
            revenue = cost = 0

        # you can select which are need to record
        record = {
            'event_id': self.curr_event_id, 
            'event_type': self.curr_event['type'], 
            'inservice': self.inservice,
            'total_revenue': self.total_revenue,
            'total_cost': self.total_cost,
            'pn_available_resource': self.pn_available_resource,
            'sfc_id': self.curr_sfc_id, 
            'result': self.success_flag, 
            'revenue': revenue,
            'cost': cost,
            'slots': self.node_slots, 
            'paths': self.edge_paths
            }
        self.records.append(record)

        if show:
            print({k:record[k] for k in ['result', 'sfc_id', 'inservice', 'pn_available_resource']})
        return record

    def save_records(self, records_dir, file_name='/records.csv'):
        r"""Save records to a csv file."""
        pd_records = pd.DataFrame(self.records)
        pd_records.to_csv(records_dir + file_name)

    @property
    def pn_available_resource(self):
        n = np.array(self.pn.get_nodes_data(['cpu', 'ram', 'rom'])).sum()
        e = np.array(self.pn.get_edges_data(['bw'])).sum()
        return n + e

if __name__ == '__main__':
    pass
    

