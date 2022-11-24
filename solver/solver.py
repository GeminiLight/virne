import abc
import copy
import os

from base import Controller, Counter


class Solver:

    def __init__(self, controller, recorder, counter, **kwargs):
        __metaclass__ = abc.ABCMeta
        self.controller = controller
        self.recorder = recorder
        self.counter = counter
        self.reusable = kwargs.get('reusable', False)
        self.verbose = kwargs.get('verbose', '1')
        self.num_arrived_v_nets = 0
        save_dir = kwargs.get('save_dir', 'save')
        solver_name = kwargs.get('solver_name', 'unknown_solver')
        run_id = kwargs.get('run_id', 'unknown_host-unknown_time')
        self.save_dir = os.path.join(save_dir, solver_name, run_id)
        # ranking strategy
        self.node_rank = None
        self.link_rank = None
        self.node_ranking_method = kwargs.get('node_ranking_method', 'order')
        self.link_ranking_method = kwargs.get('link_ranking_method', 'order')
        # node mapping
        self.matching_mathod = kwargs.get('matching_mathod', 'greedy')
        # link mapping
        self.shortest_method = kwargs.get('shortest_method', 'k_shortest')
        self.k_shortest = kwargs.get('k_shortest', 10)
        # action
        self.allow_rejection = kwargs.get('allow_rejection', False)
        self.allow_revocable = kwargs.get('allow_revocable', False)
        self.basic_config = {
            'reusable': self.reusable,
            'node_ranking_method': self.node_ranking_method,
            'link_ranking_method': self.link_ranking_method,
            'matching_mathod': self.matching_mathod,
            'shortest_method': self.shortest_method,
            'k_shortest': self.k_shortest,
            'allow_revocable': self.allow_revocable,
            'allow_revocable': self.allow_revocable
        }

    def solve(instance):
        return NotImplementedError
