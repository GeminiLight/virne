import os
import copy
import random
import numpy as np

from utils import read_setting, write_setting
from .utils import generate_data_with_distribution
from .virtual_network import VirtualNetwork


class VirtualNetworkRequestSimulator(object):
    """A simulator for generating virtual network and arrange them"""
    def __init__(self, v_sim_setting, **kwargs):
        super(VirtualNetworkRequestSimulator, self).__init__()
        self.v_sim_setting = copy.deepcopy(v_sim_setting)
        self.num_v_nets = self.v_sim_setting.get('num_v_nets', 2000)

        self.aver_arrival_rate = self.v_sim_setting['arrival_rate']['lam']

        if self.v_sim_setting['lifetime']['distribution'] == 'exponential':
            self.aver_lifetime = self.v_sim_setting['lifetime']['scale']
        elif self.v_sim_setting['lifetime']['distribution'] == 'uniform':
            self.aver_lifetime = (self.v_sim_setting['lifetime']['high'] + self.v_sim_setting['lifetime']['low']) / 2.
        else:
            raise NotImplementedError
        
        self.v_nets = []
        self.events = []

    def construct_v2event_dict(self):
        self.v2event_dict = {}
        for e_info in self.events:
            self.v2event_dict[(e_info['v_net_id'], e_info['type'])] = e_info['id']
        return self.v2event_dict
            
    @staticmethod
    def from_setting(setting):
        return VirtualNetworkRequestSimulator(setting)

    def renew(self, v_nets=True, events=True, seed=None):
        if seed is not None: 
            random.seed(seed)
            np.random.seed(seed)
        if v_nets == True:
            self.renew_v_nets()
        if events == True:
            self.renew_events()
        return self.v_nets, self.events

    def renew_v_nets(self):
        self.arrange_v_nets()
        def create_v_net(i):
            v_net = VirtualNetwork(
                node_attrs_setting=copy.deepcopy(self.v_sim_setting['node_attrs_setting']), 
                link_attrs_setting=copy.deepcopy(self.v_sim_setting['link_attrs_setting']),
                id=int(i), arrival_time=float(self.v_nets_arrival_time[i]), lifetime=float(self.v_nets_lifetime[i]))
            if 'max_latency' in self.v_sim_setting:
                v_net.set_graph_attribute('max_latency', float(self.v_nets_max_latency[i]))
            v_net.generate_topology(num_nodes=self.v_nets_size[i], **self.v_sim_setting['topology'])
            v_net.generate_attrs_data()
            return v_net
        self.v_nets = list(map(create_v_net, list(range(self.num_v_nets))))
        return self.v_nets

    def renew_events(self):
        self.events = []
        enter_list = [{'v_net_id': int(v_net.id), 'time': float(v_net.arrival_time), 'type': 1} for v_net in self.v_nets]
        leave_list = [{'v_net_id': int(v_net.id), 'time': float(v_net.arrival_time + v_net.lifetime), 'type': 0} for v_net in self.v_nets]
        event_list = enter_list + leave_list
        self.events = sorted(event_list, key=lambda e: e.__getitem__('time'))
        for i, e in enumerate(self.events): 
            e['id'] = i
        self.construct_v2event_dict()
        return self.events

    def arrange_v_nets(self):
        # length: uniform distribution
        self.v_nets_size = generate_data_with_distribution(size=self.num_v_nets, **self.v_sim_setting['v_net_size'])
        # lifetime: exponential distribution
        self.v_nets_lifetime = generate_data_with_distribution(size=self.num_v_nets, **self.v_sim_setting['lifetime'])
        # arrival_time: poisson distribution
        arrival_time_interval = generate_data_with_distribution(size=self.num_v_nets, **self.v_sim_setting['arrival_rate'])
        self.v_nets_arrival_time = np.cumsum(arrival_time_interval)
        # np.ceil(np.cumsum(np.array([-np.log(np.random.uniform()) / self.aver_arrival_rate for i in range(self.num_v_nets)]))).tolist()
        # self.v_nets_arrival_time = np.cumsum(np.random.poisson(20, self.num_v_nets))
        if 'max_latency' in self.v_sim_setting:
            self.v_nets_max_latency = generate_data_with_distribution(size=self.num_v_nets, **self.v_sim_setting['max_latency'])

    def save_dataset(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        v_nets_dir = os.path.join(save_dir, 'v_nets')
        if not os.path.exists(v_nets_dir):
            os.makedirs(v_nets_dir)
        # save v_nets
        for v_net in self.v_nets:
            v_net.to_gml(os.path.join(v_nets_dir, f'v_net-{v_net.id:05d}.gml'))
        # save events
        write_setting(self.events, os.path.join(save_dir, self.v_sim_setting['events_file_name']), mode='w+')
        self.save_setting(os.path.join(save_dir, self.v_sim_setting['setting_file_name']))

    @staticmethod
    def load_dataset(dataset_dir):
        # load setting
        if not os.path.exists(dataset_dir):
            raise ValueError(f'Find no dataset in {dataset_dir}.\nPlease firstly generating it.')
        try:
            setting_fpath = os.path.join(dataset_dir, 'v_sim_setting.yaml')
            v_sim_setting = read_setting(setting_fpath)
        except:
            setting_fpath = os.path.join(dataset_dir, 'v_sim_setting.json')
            v_sim_setting = read_setting(setting_fpath)
        v_net_simulator = VirtualNetworkRequestSimulator.from_setting(v_sim_setting)
        # load v_nets
        v_net_fnames_list = os.listdir(os.path.join(dataset_dir, 'v_nets'))
        v_net_fnames_list.sort()
        for v_net_fname in v_net_fnames_list:
            v_net = VirtualNetwork.from_gml(os.path.join(dataset_dir, 'v_nets', v_net_fname))
            v_net_simulator.v_nets.append(v_net)
        # load events
        events = read_setting(os.path.join(dataset_dir, v_sim_setting['events_file_name']))
        v_net_simulator.events = events
        v_net_simulator.construct_v2event_dict()
        return v_net_simulator

    def save_setting(self, fpath):
        write_setting(self.v_sim_setting, fpath, mode='w+')


if __name__ == '__main__':
    pass
