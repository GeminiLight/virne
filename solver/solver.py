import abc
import copy
import os

from base import Controller


class Solver:

    def __init__(self, name, reusable=False, verbose=1, **kwargs):
        __metaclass__ = abc.ABCMeta
        self.name = name
        self.reusable = reusable
        self.verbose = verbose
        self.num_arrived_vns = 0
        self.controller = Controller()
        self.node_rank = None
        self.edge_rank = None
        save_dir = kwargs.get('save_dir', 'save')
        solver_name = kwargs.get('solver_name', 'unknown_solver')
        host_time = f'{kwargs.get("host_name", "unknown_host")}-{kwargs.get("run_time", "unknown_time")}'
        self.save_dir = os.path.join(save_dir, solver_name, host_time)
        # node mapping
        self.matching_mathod = kwargs.get('matching_mathod', 'greedy')
        # link mapping
        self.shortest_method = kwargs.get('shortest_method', 'k_shortest')
        self.k_shortest = kwargs.get('k_shortest', 10)


    @classmethod
    def from_config(cls, config):
        if not isinstance(config, dict): config = vars(config)
        config = copy.deepcopy(config)
        reusable = config.pop('reusable', False)
        verbose = config.pop('verbose', 1)
        return cls(reusable=reusable, verbose=verbose, **config)

    def solve(obs):
        return NotImplementedError

    def learn(self, *args, **kwargs):
        return

    def load_model(self, *args, **kwargs):
        return

    def save_model(self, *args, **kwargs):
        return