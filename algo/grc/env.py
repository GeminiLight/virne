import os
import sys
import abc
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

file_path_dir = os.path.abspath('.')
if file_path_dir not in sys.path:
    sys.path.append(file_path_dir)

from data.physical_network import PhysicalNetwork
from algo import Environment

class GRCEnv(Environment):
    """The environment for GRC appoarch."""
    def __init__(self, **kwargs):
        super(GRCEnv, self).__init__(**kwargs)

    def step(self, flag, slots, paths):
        """Environment tackles leave events or records history."""
        self.success_flag = flag
        
        if flag:
            self.success += 1
            self.inservice += 1
            self.node_slots = slots
            self.edge_paths = paths
            self.curr_sfc_cost = self.calculate_curr_sfc_cost()
            self.total_revenue += self.curr_sfc_revenue
            self.total_cost += self.curr_sfc_cost
        else:
            self.curr_sfc_revenue = 0
            self.pn = copy.deepcopy(self.pn_backup)

        return True


if __name__ == '__main__':
    # new a env
    pass
    