import os
import csv
import copy
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict


###-----------###
#   Objective   #
###-----------###


class Recorder:
    r"""Record the environment's states and solutions' information during the deployment process"""
    def __init__(self, counter, summary_dir='records/', save_dir='records/', if_temp_save_records=True, **kwargs):
        self.counter = counter
        self.summary_dir = summary_dir
        self.save_dir = save_dir
        self.if_temp_save_records = if_temp_save_records

        self.record_dir = os.path.join(self.save_dir, kwargs.get('solver_name'), kwargs.get('run_id'), 'records')
        if not os.path.exists(self.record_dir):
            try:
                os.makedirs(self.record_dir)
            except:
                pass
        self.reset()

    def reset(self):
        self.curr_record = {}
        self.memory = []
        self.v_net_event_dict = {}  # for querying the record of v_net
        self.p_net_nodes_for_v_net_dict = defaultdict(list)
        self.state = {
            'v_net_count': 0, 
            'success_count': 0, 
            'inservice_count': 0,
            'total_revenue': 0, 
            'total_cost': 0, 
            'total_r2c': 0,
            'total_time_revenue': 0,
            'total_time_cost': 0,
            'num_running_p_net_nodes': 0
        }
        if self.if_temp_save_records:
            suffixes = 0
            available_fname = False
            while not available_fname:
                temp_save_path = os.path.join(self.record_dir , f'temp-{suffixes}.csv')
                suffixes += 1
                if not os.path.exists(temp_save_path):
                    available_fname = True
            self.temp_save_path = temp_save_path
            self.written_temp_header = False

    def count_init_p_net_info(self, p_net):
        self.init_p_net_info = {}
        self.init_p_net_info['p_net_available_resource'] = self.counter.calculate_sum_network_resource(p_net)
        self.init_p_net_info['p_net_node_available_resource'] = self.counter.calculate_sum_network_resource(p_net, link=False)
        self.init_p_net_info['p_net_link_available_resource'] = self.counter.calculate_sum_network_resource(p_net, node=False)

    def update_state(self, info_dict):
        self.state.update(info_dict)
    # def ready(self, event):
    #     self.state['event_id'] = event['id']
    #     self.state['event_time'] = event['time']
    #     self.state['event_type'] = event['type']
    
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

    def count_state(cls, v_net, p_net, solution):
        cls.state['p_net_available_resource'] = cls.counter.calculate_sum_network_resource(p_net)
        cls.state['p_net_node_available_resource'] = cls.counter.calculate_sum_network_resource(p_net, link=False)
        cls.state['p_net_link_available_resource'] = cls.counter.calculate_sum_network_resource(p_net, node=False)
        cls.state['p_net_node_resource_utilization'] = 1. - (cls.state['p_net_node_available_resource'] / cls.init_p_net_info['p_net_node_available_resource'])
        cls.state['p_net_link_resource_utilization'] = 1. - (cls.state['p_net_link_available_resource'] / cls.init_p_net_info['p_net_link_available_resource'])
        # Leave event
        if cls.state['event_type'] == 0:
            if cls.get_record(v_net_id=solution['v_net_id'])['result']:
                cls.state['inservice_count'] -= 1
                v_net_id = solution['v_net_id']
                for v_node_id, p_node_id in solution['node_slots'].items():
                    cls.p_net_nodes_for_v_net_dict[p_node_id].remove(v_net_id)
                cls.state['num_running_p_net_nodes'] = len(cls.get_running_p_net_nodes())
        # Enter event
        elif cls.state['event_type'] == 1:
            cls.v_net_event_dict[solution['v_net_id']] = cls.state['event_id']
            cls.state['v_net_count'] += 1
            # Success
            if solution['result']:
                cls.state['success_count'] += 1
                cls.state['inservice_count'] += 1
                cls.state['total_revenue'] += solution['v_net_revenue']
                cls.state['total_cost'] += solution['v_net_cost']
                cls.state['total_r2c'] = cls.state['total_revenue'] / cls.state['total_cost'] if cls.state['total_cost'] else 0
                cls.state['total_time_revenue'] += solution['v_net_time_revenue']
                cls.state['total_time_cost'] += solution['v_net_time_cost']
                v_net_id = solution['v_net_id']
                for v_node_id, p_node_id in solution['node_slots'].items():
                    cls.p_net_nodes_for_v_net_dict[p_node_id].append(v_net_id)
                cls.state['num_running_p_net_nodes'] = len(cls.get_running_p_net_nodes())
        else:
            raise NotImplementedError

    def count(cls, v_net, p_net, solution):
        r"""Count the information of environment state and v_net solution."""
        # Leave event
        if cls.state['event_type'] == 0:
            solution_info = solution.to_dict()
        # Enter event
        elif cls.state['event_type'] == 1:
            solution_info = cls.counter.count_solution(v_net, solution)
        else:
            raise NotImplementedError
            
        cls.count_state(v_net, p_net, solution)
        record = {**cls.state, **solution_info}
        return record

    def get_running_p_net_nodes(cls):
        return [p_net_nodes for p_net_nodes, v_nets_list in cls.p_net_nodes_for_v_net_dict.items() if len(v_nets_list) != 0]

    def get_record(self, event_id=None, v_net_id=None):
        r"""Get the record of the service function chain `v_net_id`."""
        if event_id is not None: event_id = event_id
        elif v_net_id is not None: event_id = self.v_net_event_dict[v_net_id]
        else: event_id = self.state['event_id']
        return self.memory[int(event_id)]

    def display_record(self, record, display_items=['result', 'v_net_id', 'v_net_cost', 'v_net_revenue', 'p_net_available_resource', 'total_revenue', 'total_cost', 'description'], extra_items=[]):
        display_items = display_items + extra_items
        print(''.join([f'{k}: {v}\n' for k, v in record.items() if k in display_items]))

    def temp_save_record(self, record):
        with open(self.temp_save_path, 'a+', newline='') as f:
            writer = csv.writer(f)
            if not self.written_temp_header:
                writer.writerow(record.keys())
            writer.writerow(record.values())
        self.written_temp_header = True
        
    def save_records(self, fname):
        r"""Save records to a csv file."""
        save_path = os.path.join(self.record_dir, fname)
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

