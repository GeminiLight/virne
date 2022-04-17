import os
import csv
import copy
from typing import Dict
import numpy as np
import pandas as pd
from collections import defaultdict


###-----------###
#   Objective   #
###-----------###


class Recorder:
    r"""Record the environment's states and solutions' information during the deployment process"""
    def __init__(self, summary_dir='records/', save_dir='records/', if_temp_save_records=True, node_resource_price=0.33, edge_resource_price=1.):
        self.counter = Counter()
        self.summary_dir = summary_dir
        self.save_dir = save_dir
        self.if_temp_save_records = if_temp_save_records
        self.curr_record = {}
        self.node_resource_price = node_resource_price
        self.edge_resource_price = edge_resource_price
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        # self.reset()

    def reset(self):
        self.memory = []
        self.vn_event_dict = {}  # for querying the record of vn
        self.pn_nodes_for_vn_dict = defaultdict(list)
        self.state = {
            'vn_count': 0, 
            'success_count': 0, 
            'inservice_count': 0,
            'total_revenue': 0, 
            'total_cost': 0, 
            'total_r2c': 0,
            'total_time_revenue': 0,
            'total_time_cost': 0,
            'num_running_pn_nodes': 0
        }
        if self.if_temp_save_records:
            suffixes = 0
            available_fname = False
            while not available_fname:
                temp_save_path = os.path.join(self.save_dir , f'temp-{suffixes}.csv')
                suffixes += 1
                if not os.path.exists(temp_save_path):
                    available_fname = True
            self.temp_save_path = temp_save_path
            if not os.path.exists(self.temp_save_path):
                open(self.temp_save_path, 'w')
            self.written_temp_header = False

    def count_init_pn_info(self, pn):
        self.init_pn_info = {}
        self.init_pn_info['pn_available_resource'] = self.counter.calculate_sum_network_resource(pn)
        self.init_pn_info['pn_node_available_resource'] = self.counter.calculate_sum_network_resource(pn, edge=False)
        self.init_pn_info['pn_edge_available_resource'] = self.counter.calculate_sum_network_resource(pn, node=False)

    def ready(self, event):
        self.state['event_id'] = event['id']
        self.state['event_time'] = event['time']
        self.state['event_type'] = event['type']
    
    def add_info(self, info_dict, **kwargs):
        self.curr_record.update(info_dict)
        self.curr_record.update(kwargs)

    def add_record(self, record, extra_info={}, **kwargs):
        self.curr_record = {}
        self.curr_record.update(record)
        self.curr_record.update(extra_info)
        self.curr_record.update(kwargs)
        self.memory.append(copy.deepcopy(self.curr_record))
        if self.if_temp_save_records: self.temp_save_record(self.curr_record)
        return self.curr_record

    def count_state(cls, vn, pn, solution):
        cls.vn_event_dict[solution['vn_id']] = cls.state['event_id']
        cls.state['pn_available_resource'] = cls.counter.calculate_sum_network_resource(pn)
        cls.state['pn_node_available_resource'] = cls.counter.calculate_sum_network_resource(pn, edge=False)
        cls.state['pn_edge_available_resource'] = cls.counter.calculate_sum_network_resource(pn, node=False)
        cls.state['pn_node_resource_utilization'] = 1. - (cls.state['pn_node_available_resource'] / cls.init_pn_info['pn_node_available_resource'])
        cls.state['pn_edge_resource_utilization'] = 1. - (cls.state['pn_edge_available_resource'] / cls.init_pn_info['pn_edge_available_resource'])
        # Leave event
        if cls.state['event_type'] == 0:
            if cls.get_record(solution['vn_id'])['result']:
                cls.state['inservice_count'] -= 1
                vn_id = solution['vn_id']
                for vnf_id, pn_node_id in solution['node_slots'].items():
                    cls.pn_nodes_for_vn_dict[pn_node_id].remove(vn_id)
                cls.state['num_running_pn_nodes'] = len(cls.get_running_pn_nodes())
        # Enter event
        elif cls.state['event_type'] == 1:
            cls.state['vn_count'] += 1
            # Success
            if solution['result']:
                cls.state['success_count'] += 1
                cls.state['inservice_count'] += 1
                cls.state['total_revenue'] += solution['vn_revenue']
                cls.state['total_cost'] += solution['vn_cost']
                cls.state['total_r2c'] = cls.state['total_revenue'] / cls.state['total_cost'] if cls.state['total_cost'] else 0
                cls.state['total_time_revenue'] += solution['vn_time_revenue']
                cls.state['total_time_cost'] += solution['vn_time_cost']
                vn_id = solution['vn_id']
                for vnf_id, pn_node_id in solution['node_slots'].items():
                    cls.pn_nodes_for_vn_dict[pn_node_id].append(vn_id)
                cls.state['num_running_pn_nodes'] = len(cls.get_running_pn_nodes())
        else:
            raise NotImplementedError

    def count(cls, vn, pn, solution):
        r"""Count the information of environment state and vn solution."""
        # Leave event
        if cls.state['event_type'] == 0:
            solution_info = solution.to_dict()
        # Enter event
        elif cls.state['event_type'] == 1:
            solution_info = cls.counter.count_solution(vn, solution)
        else:
            raise NotImplementedError
            
        cls.count_state(vn, pn, solution)
        record = {**cls.state, **solution_info}
        return record

    def get_running_pn_nodes(cls):
        return [pn_nodes for pn_nodes, vns_list in cls.pn_nodes_for_vn_dict.items() if len(vns_list) != 0]

    def get_record(self, event_id=None, vn_id=None):
        r"""Get the record of the service function chain `vn_id`."""
        if event_id is not None: event_id = event_id
        elif vn_id is not None: event_id = self.vn_event_dict[vn_id]
        else: event_id = self.state['event_id']
        return self.memory[int(event_id)]

    def display_record(self, record, display_items=['result', 'vn_id', 'vn_cost', 'vn_revenue', 'pn_available_resource', 'total_revenue', 'total_cost', 'description'], extra_items=[]):
        display_items = display_items + extra_items
        print(''.join([f'{k}: {v}\n' for k, v in record.items() if k in display_items]))

    def temp_save_record(self, record):
        with open(self.temp_save_path, 'a+', newline='') as f:  # Just use 'w' mode in 3.x
            writer = csv.writer(f)
            if not self.written_temp_header:
                writer.writerow(record.keys())
            writer.writerow(record.values())
        self.written_temp_header = True
        
    def save_records(self, fname):
        r"""Save records to a csv file."""
        save_path = os.path.join(self.save_dir, fname)
        pd_records = pd.DataFrame(self.memory)
        pd_records.to_csv(save_path, index=False)
        try:
            os.remove(self.temp_save_path)
        except:
            pass
        return save_path

    ### summary ###
    def summary_records(self, records):
        return self.counter.summary_records(records)

    def save_summary(self, summary_info, fname='global_summary.csv'):
        summary_path = os.path.join(self.summary_dir,  fname)
        head = None if os.path.exists(summary_path) else list(summary_info.keys())
        with open(summary_path, 'a+', newline='') as csv_file:
            writer = csv.writer(csv_file, dialect='excel', delimiter=',')
            if head is not None: writer.writerow(head)
            writer.writerow(list(summary_info.values()))
        return summary_path


class Counter(object):

    @staticmethod
    def count_partial_solution(vn, solution):
        n_attrs = vn.get_node_attrs(['resource'])
        e_attrs = vn.get_edge_attrs(['resource'])
        # node revenue
        vn_node_revenue = 0
        for nid in solution['node_slots'].keys():
            vn_node_revenue += sum([vn.nodes[nid][n_attr.name] for n_attr in n_attrs])
        vn_edge_revenue = 0
        vn_path_cost = 0
        # edge revenue
        for edge, path in solution['edge_paths'].items():
            one_revenue = sum([vn.edges[edge][e_attr.name] for e_attr in e_attrs])
            vn_edge_revenue += one_revenue
            vn_path_cost += one_revenue * (len(path) - 2)

        solution['vn_node_revenue'] = vn_node_revenue
        solution['vn_edge_revenue'] = vn_edge_revenue

        solution['vn_revenue'] = vn_node_revenue + vn_edge_revenue
        solution['vn_path_cost'] = vn_path_cost
        solution['vn_node_cost'] = vn_node_revenue
        solution['vn_edge_cost'] = vn_edge_revenue + vn_path_cost
        solution['vn_cost'] = solution['vn_revenue'] + vn_path_cost
        solution['vn_rc_ratio'] = solution['vn_revenue'] / solution['vn_cost'] if solution['vn_cost'] != 0 else 0
        return solution.__dict__

    @staticmethod
    def count_solution(vn, solution) -> Dict:
        # Success
        if solution['result']:
            solution['vn_node_revenue'] = Counter.calculate_sum_node_resource(vn)
            solution['vn_edge_revenue'] = Counter.calculate_sum_edge_resource(vn)
            solution['vn_revenue'] = solution['vn_node_revenue'] + solution['vn_edge_revenue']
            solution['vn_path_cost'] = Counter.calculate_vn_path_cost(vn, solution)
            solution['vn_cost'] = solution['vn_revenue'] + solution['vn_path_cost']
            solution['vn_rc_ratio'] = solution['vn_revenue'] / solution['vn_cost'] if solution['vn_cost'] != 0 else 0
        # Faliure
        else:
            solution['vn_node_revenue'] = 0
            solution['vn_edge_revenue'] = 0
            solution['vn_revenue'] = 0
            solution['vn_path_cost'] = 0
            solution['vn_cost'] = 0
            solution['vn_rc_ratio'] = 0
            # solution['node_slots'] = {}
            # solution['edge_paths'] = {}
        solution['vn_time_revenue'] = solution['vn_revenue'] * vn.lifetime
        solution['vn_time_cost'] = solution['vn_cost'] * vn.lifetime
        solution['vn_time_rc_ratio'] = solution['vn_rc_ratio'] * vn.lifetime
        return solution.__dict__

    @staticmethod
    def summary_records(records):
        if isinstance(records, list):
            records = pd.DataFrame(records)
        elif isinstance(records, pd.DataFrame):
            pass
        else:
            raise TypeError
        summary_info = {}

        # ac rate
        summary_info['success_count'] = records.iloc[-1]['success_count']
        summary_info['acceptance_rate'] = records.iloc[-1]['success_count'] / records.iloc[-1]['vn_count']

        # revenue / cost
        summary_info['total_cost'] = records.iloc[-1]['total_cost']
        summary_info['total_revenue'] = records.iloc[-1]['total_revenue']
        summary_info['total_time_cost'] = records.iloc[-1]['total_time_cost']
        summary_info['total_time_revenue'] = records.iloc[-1]['total_time_revenue']
        summary_info['running_time'] = records[records['event_type']==1].iloc[-1]['arrival_time']
        summary_info['long_term_aver_revenue'] = summary_info['total_revenue'] / summary_info['running_time']
        summary_info['total_running_time'] = records.iloc[-1]['arrival_time']

        # rc ratio
        summary_info['rc_ratio'] = records.iloc[-1]['total_revenue'] / records.iloc[-1]['total_cost']
        summary_info['time_rc_ratio'] = records.iloc[-1]['total_time_revenue'] / records.iloc[-1]['total_time_cost']

        # other
        summary_info['min_pn_available_resource'] = records.loc[:, 'pn_available_resource'].min()
        summary_info['min_pn_node_available_resource'] = records.loc[:, 'pn_node_available_resource'].min()
        summary_info['min_pn_edge_available_resource'] = records.loc[:, 'pn_edge_available_resource'].min()
        summary_info['max_inservice_count'] = records.loc[:, 'inservice_count'].max()
        
        # rl reward
        if 'cumulative_reward' in records.columns:
            cumulative_rewards = records.loc[:, 'cumulative_reward'].dropna()
            summary_info['cumulative_reward'] = cumulative_rewards.iloc[-1]
        else:
            summary_info['cumulative_reward'] = 0
        return summary_info

    @staticmethod
    def summary_csv(fpath):
        records = pd.read_csv(fpath, header=0)
        summary_info = Counter.summary_records(records)
        return summary_info

    @staticmethod
    def calculate_sum_network_resource(network, node=True, edge=True):
        n = np.array(network.get_node_attrs_data(network.get_node_attrs('resource'))).sum() if node else 0
        e = np.array(network.get_edge_attrs_data(network.get_edge_attrs('resource'))).sum() if edge else 0
        return n + e

    @staticmethod
    def calculate_sum_node_resource(network):
        n = np.array(network.get_node_attrs_data(network.get_node_attrs('resource'))).sum()
        return n

    @staticmethod
    def calculate_sum_edge_resource(network):
        e = np.array(network.get_edge_attrs_data(network.get_edge_attrs('resource'))).sum()
        return e

    @staticmethod
    def calculate_vn_revenue(vn, solution=None):
        r"""Calculate the deployment cost of current vn according to `edge paths`."""
        return Counter.calculate_sum_network_resource(vn)

    @staticmethod
    def calculate_vn_path_cost(vn, solution=None):
        r"""Calculate the deployment cost of current vn according to `edge paths`."""
        vn_path_cost = 0
        e_attrs = vn.get_edge_attrs('resource')
        for edge, path in solution['edge_paths'].items():
            revenue = sum([vn.edges[edge][e_attr.name] for e_attr in e_attrs])
            vn_path_cost += revenue * (len(path) - 2)
        return vn_path_cost

    @staticmethod
    def calculate_vn_cost(vn, solution=None):
        r"""Calculate the deployment cost of current vn according to `edge paths`."""
        vn_revenue = Counter.calculate_vn_revenue(vn, solution)
        vn_path_cost = Counter.calculate_vn_path_cost(vn, solution)
        return vn_revenue + vn_path_cost

class ClassDict(object):
    def __init__(self):
        super(ClassDict, self).__init__()

    def update(self, *args, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    @classmethod
    def from_dict(cls, dict):
        cls.__dict__ = dict
        return cls

    def to_dict(self):
        return self.__dict__

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key, None)
        elif isinstance(key, int):
            return super().__getitem__(key)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)


class Solution(ClassDict):
    def __init__(self, vn):
        super(Solution, self).__init__()
        # if isinstance(vn, int):
        #     self.vn_id = vn
        # elif isinstance(vn, VirtualNetwork):
        self.vn_id = vn.id
        self.lifetime = vn.lifetime
        self.arrival_time = vn.arrival_time
        # else:
            # raise TypeError('')
        self.reset()

    def reset(self):
        self.result = False
        self.node_slots = {}
        self.edge_paths = {}
        self.vn_cost = 0
        self.vn_revenue = 0
        self.vn_node_revenue = 0
        self.vn_edge_revenue = 0
        self.vn_node_cost = 0
        self.vn_edge_cost = 0
        self.vn_path_cost = 0
        self.vn_rc_ratio = 0
        self.vn_time_cost = 0
        self.vn_time_revenue = 0
        self.vn_time_rc_ratio = 0
        self.description = ''
        self.place_result = True
        self.route_result = True
        self.early_rejection = False
    