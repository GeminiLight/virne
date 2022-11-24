import pprint
from collections import OrderedDict


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

    def __init__(self, v_net):
        super(Solution, self).__init__()
        self.v_net_id = v_net.id
        self.v_net_lifetime = v_net.lifetime
        self.v_net_arrival_time = v_net.arrival_time
        self.v_net_num_nodes = v_net.num_nodes
        self.v_net_num_egdes = v_net.num_links
        self.reset()

    def reset(self):
        self.result = False
        self.node_slots = OrderedDict()
        self.link_paths = OrderedDict()
        self.node_slots_info = OrderedDict()
        self.link_paths_info = OrderedDict()
        self.v_net_cost = 0
        self.v_net_revenue = 0
        self.v_net_demand = 0
        self.v_net_node_demand = 0
        self.v_net_link_demand = 0
        self.v_net_node_revenue = 0
        self.v_net_link_revenue = 0
        self.v_net_node_cost = 0
        self.v_net_link_cost = 0
        self.v_net_path_cost = 0
        self.v_net_r2c_ratio = 0
        self.v_net_time_cost = 0
        self.v_net_time_revenue = 0
        self.v_net_time_rc_ratio = 0
        self.description = ''
        self.violation = 0
        self.current_violation = 0
        self.total_place_violation = 0
        self.total_route_violation = 0
        self.place_result = True
        self.route_result = True
        self.early_rejection = False
        self.revoke_times = 0
        self.selected_actions = []

    def is_feasible(self):
        return self.result and self.violation <= 0

    def display(self):
        pprint.pprint(self.__dict__)