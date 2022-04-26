import os
import copy
import numpy as np

from .utils import read_setting, write_setting, generate_data_with_distribution
from .virtual_network import VirtualNetwork


class VNSimulator(object):
    """A simulator for generating virtual network and arrange them"""
    def __init__(self, vns_setting, **kwargs):
        super(VNSimulator, self).__init__()
        self.vns_setting = copy.deepcopy(vns_setting)
        self.num_vns = self.vns_setting.get('num_vns', 2000)
        self.topology = self.vns_setting.get('topology', {'type': 'random', 'random_prob': 0.5})
        self.max_vn_size = self.vns_setting['vn_size']['high']
        self.min_vn_size = self.vns_setting['vn_size']['low']
        self.aver_arrival_rate = self.vns_setting['arrival_rate']['lam']
        self.aver_lifetime = self.vns_setting['lifetime']['scale']
        
        self.vns = []
        self.events = []

    @staticmethod
    def from_setting(setting):
        return VNSimulator(setting)

    def renew(self, vns=True, events=True):
        if vns == True:
            self.renew_vns()
        if events == True:
            self.renew_events()
        return self.vns, self.events

    def renew_vns(self):
        self.vns = []
        self.arrange_vns()
        for i in range(self.num_vns):
            vn = VirtualNetwork(
                node_attrs_setting=copy.deepcopy(self.vns_setting['node_attrs_setting']), 
                edge_attrs_setting=copy.deepcopy(self.vns_setting['edge_attrs_setting']),
                id=int(i), arrival_time=float(self.vns_arrival_time[i]), lifetime=float(self.vns_lifetime[i]))
            vn.generate_topology(num_nodes=self.vns_size[i], **self.topology)
            vn.generate_attrs_data()
            self.vns.append(vn)
        return self.vns

    def renew_events(self):
        self.events = []
        enter_list = [{'vn_id': int(vn.id), 'time': float(vn.arrival_time), 'type': 1} for vn in self.vns]
        leave_list = [{'vn_id': int(vn.id), 'time': float(vn.arrival_time + vn.lifetime), 'type': 0} for vn in self.vns]
        event_list = enter_list + leave_list
        self.events = sorted(event_list, key=lambda e: e.__getitem__('time'))
        for i, e in enumerate(self.events): 
            e['id'] = i
        return self.events

    def arrange_vns(self):
        # length: uniform distribution
        self.vns_size = generate_data_with_distribution(size=self.num_vns, **self.vns_setting['vn_size'])
        # lifetime: exponential distribution
        self.vns_lifetime = generate_data_with_distribution(size=self.num_vns, **self.vns_setting['lifetime'])
        # arrival_time: poisson distribution
        arrival_time_interval = generate_data_with_distribution(size=self.num_vns, **self.vns_setting['arrival_rate'])
        self.vns_arrival_time = np.cumsum(arrival_time_interval)
        # np.ceil(np.cumsum(np.array([-np.log(np.random.uniform()) / self.aver_arrival_rate for i in range(self.num_vns)]))).tolist()
        # self.vns_arrival_time = np.cumsum(np.random.poisson(20, self.num_vns))
    
    def save_dataset(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        vns_dir = os.path.join(save_dir, 'vns')
        if not os.path.exists(vns_dir):
            os.makedirs(vns_dir)
        # save vns
        for vn in self.vns:
            vn.to_gml(os.path.join(vns_dir, f'vn-{vn.id:05d}.gml'))
        # save events
        write_setting(self.events, os.path.join(save_dir, self.vns_setting['events_file_name']), mode='w+')
        self.save_setting(os.path.join(save_dir, self.vns_setting['setting_file_name']))

    @staticmethod
    def load_dataset(dataset_dir):
        # load setting
        if not os.path.exists(dataset_dir):
            raise ValueError(f'Find no dataset in {dataset_dir}.\nPlease firstly generating it.')
        try:
            setting_fpath = os.path.join(dataset_dir, 'vns_setting.yaml')
            vns_setting = read_setting(setting_fpath)
        except:
            setting_fpath = os.path.join(dataset_dir, 'vns_setting.json')
            vns_setting = read_setting(setting_fpath)
        vn_simulator = VNSimulator.from_setting(vns_setting)
        # load vns
        vn_fnames_list = os.listdir(os.path.join(dataset_dir, 'vns'))
        vn_fnames_list.sort()
        for vn_fname in vn_fnames_list:
            vn = VirtualNetwork.from_gml(os.path.join(dataset_dir, 'vns', vn_fname))
            vn_simulator.vns.append(vn)
        # load events
        events = read_setting(os.path.join(dataset_dir, vns_setting['events_file_name']))
        vn_simulator.events = events
        return vn_simulator

    def save_setting(self, fpath):
        # setting = {}
        # for k, v in self.__dict__.items():
        #     if k not in ['events', 'vns', 'vns_size', 'vns_lifetime', 'vns_arrival_time']:
        #         setting[k] = v
        write_setting(self.vns_setting, fpath, mode='w+')


if __name__ == '__main__':
    pass
