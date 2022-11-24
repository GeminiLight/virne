import os
import copy
import time
import numpy as np
from pprint import pprint

from utils import get_p_net_dataset_dir_from_setting, get_v_nets_dataset_dir_from_setting
from data import PhysicalNetwork, VirtualNetworkRequestSimulator
from .recorder import Recorder
from .solution import Solution


class Environment:
    r"""A general environment for various solvers based on heuristics and RL"""
    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, **kwargs):
        self.p_net = p_net
        self.init_p_net = copy.deepcopy(p_net)
        self.v_net_simulator = v_net_simulator
        self.num_v_nets = self.v_net_simulator.num_v_nets
        self.num_events = self.num_v_nets * 2
        self.controller = controller
        self.recorder = recorder
        self.counter = counter
        self.verbose = kwargs.get('verbose', 1)
        self.p_net_dataset_dir = kwargs.get('p_net_dataset_dir', 'unknown_p_net_dataset_dir')
        self.v_nets_dataset_dir = kwargs.get('v_nets_dataset_dir', 'unknown_v_nets_dataset_dir')
        self.renew_v_net_simulator = kwargs.get('renew_v_net_simulator', False)

        self.solver_name = kwargs.get('solver_name', 'unknown_slover')
        self.run_id = kwargs.get('run_id', 'unknown_device-unknown_run_time')
        self.seed = kwargs.get('seed', None)
        self.if_save_records = kwargs.get('if_save_records', True)
        self.summary_file_name = kwargs.get('summary_file_name', 'global_summary.csv')

        # ranking strategy
        self.node_ranking_method = kwargs.get('node_ranking_method', 'order')
        self.link_ranking_method = kwargs.get('link_ranking_method', 'order')
        # node mapping
        self.matching_mathod = kwargs.get('matching_mathod', 'greedy')
        # link mapping
        self.shortest_method = kwargs.get('shortest_method', 'k_shortest')
        self.k_shortest = kwargs.get('k_shortest', 10)

        self.extra_summary_info = {}
        self.extra_record_info = {}

    # @classmethod
    # def from_config(cls, config):
    #     r"""Create an environment following the settings in config."""
    #     if not isinstance(config, dict): config = vars(config)
    #     config = copy.deepcopy(config)

    #     verbose = config.pop('verbose', 1)
    #     p_net_setting = config.get('p_net_setting')
    #     v_sim_setting = config.get('v_sim_setting')
    #     p_net_dataset_dir = get_p_net_dataset_dir_from_setting(p_net_setting)
    #     config['p_net_dataset_dir'] = p_net_dataset_dir
    #     # create p_net and v_sim
    #     if os.path.exists(p_net_dataset_dir):
    #         p_net = PhysicalNetwork.load_dataset(p_net_dataset_dir)
    #         print(f'Load Physical Network from {p_net_dataset_dir}') if verbose >= 1 else None
    #     else:
    #         p_net = PhysicalNetwork.from_setting(p_net_setting)
    #         print(f'Generate Physical Network from setting') if verbose >= 1 else None
    #     v_net_simulator = VirtualNetworkRequestSimulator.from_setting(v_sim_setting)
    #     recorder = Recorder(**config)
    #     return cls(p_net, v_net_simulator, controller, recorder, counter, **config)

    def ready(self, event_id=0):
        r"""Ready to attempt to execuate the current events."""
        self.curr_event = self.v_net_simulator.events[event_id]
        self.v_net = self.v_net_simulator.v_nets[int(self.curr_event['v_net_id'])]
        self.solution = Solution(self.v_net)
        self.p_net_backup = copy.deepcopy(self.p_net) if self.curr_event['type'] == 1 else None
        self.recorder.update_state({
            'event_id': self.curr_event['id'],
            'event_type': self.curr_event['type'],
            'event_time': self.curr_event['time'],
        })
        # self.recorder.ready(self.curr_event)
        if self.verbose >= 2:
            print(f"\nEvent: id={event_id}, type={self.curr_event['type']}")
            print(f"{'-' * 30}")

    def reset(self, seed=None):
        r"""Reset the environment."""
        seed = seed if seed is not None else self.seed

        self.p_net = copy.deepcopy(self.init_p_net)
        self.recorder.reset()
        self.recorder.count_init_p_net_info(self.p_net)
        if self.recorder.if_temp_save_records and self.verbose >= 1:
            print(f'temp save record in {self.recorder.temp_save_path}\n')

        self.v_nets_dataset_dir = get_v_nets_dataset_dir_from_setting(self.v_net_simulator.v_sim_setting)
        if self.renew_v_net_simulator:
            self.v_net_simulator.renew(v_nets=True, events=True, seed=seed)
            print(f'Generate virtual networks with seed {seed}') if self.verbose >= 1 else None
        elif os.path.exists(self.v_nets_dataset_dir):
            self.v_net_simulator = self.v_net_simulator.load_dataset(self.v_nets_dataset_dir)
            print(f'Load virtual networks from {self.v_nets_dataset_dir}') if self.verbose >= 1 else None
        else: 
            self.v_net_simulator.renew(v_nets=True, events=True, seed=seed)
            print(f'\n*** Generate virtual networks with seed {seed}')
        self.cumulative_reward = 0
        self.start_run_time = time.time()
        self.ready(event_id=0)
        return self.get_observation()

    def step(self, action):
        return NotImplementedError

    def compute_reward(self):
        return NotImplementedError

    def get_observation(self):
        return NotImplementedError

    def render(self, mode="human"):
        return NotImplementedError

    def release(self):
        r"""Release occupied resources when a VN leaves PN."""
        solution = self.recorder.get_record(v_net_id=self.v_net['id'])
        self.controller.release(self.v_net, self.p_net, solution)
        self.solution['description'] = 'Leave Event'
        record = self.count_and_add_record()
        return record

    def get_failure_reason(self, solution):
        if solution['early_rejection']:
            return 'reject'
        if not solution['place_result']:
            return 'place'
        elif not solution['route_result']:
            return 'route'
        else:
            return 'unknown'

    def rollback_for_failure(self, reason='place'):
        r"""Restore the state of the physical network and record the reason of failure."""
        # self.solution.reset()
        self.p_net = copy.deepcopy(self.p_net_backup)
        if reason in ['unknown', -1]:
            self.solution['description'] = 'Unknown Reason'
        if reason in ['reject', 0]:
            self.solution['description'] = 'Early Rejection'
            self.solution['early_rejection'] = True
        elif reason in ['place', 1]:
            self.solution['description'] = 'Place Failure'
            self.solution['place_result'] = False
        elif reason in ['route', 2]:
            self.solution['description'] = 'Route Failure'
            self.solution['route_result'] = False
        else:
            return NotImplementedError

    def transit_obs(self):
        """Automatically execute the leave events 
            until the next enter event comes or episode has done.
        
        Return:
            done (bool): whether the current episode has done
        """
        # Leave events transition
        while True:
            next_event_id = int(self.curr_event['id'] + 1)
            # episode finished
            if next_event_id > self.num_events - 1:
                return True
            self.ready(next_event_id)
            if self.curr_event['type'] == 0:
                record = self.release()
            else:
                return False

    @property
    def selected_p_net_nodes(self):
        return list(self.solution['node_slots'].values())

    @property
    def placed_v_net_nodes(self):
        return list(self.solution['node_slots'].keys())

    ### recorder ###
    def add_record(self, record, extra_info={}):
        record = self.recorder.add_record(record, extra_info)
        if self.verbose >= 2:
            self.display_record(record, extra_items=list(extra_info.keys()))
        return record

    def count_and_add_record(self, extra_info={}):
        record = self.recorder.count(self.v_net, self.p_net, self.solution)
        record = self.add_record(record, extra_info)
        return record

    def display_record(self, record, display_items=['result', 'v_net_id', 'v_net_cost', 'v_net_revenue', 
                        'p_net_available_resource', 'total_revenue', 'total_cost', 'description'], extra_items=[]):
        display_items = display_items + extra_items
        print(''.join([f'{k}: {v}\n' for k, v in record.items() if k in display_items]))

    def summary_records(self, extra_summary_info={}, summary_file_name=None, record_file_name=None):
        start_run_time = time.strftime('%Y%m%dT%H%M%S', time.localtime(self.start_run_time))
        if summary_file_name is None:
            summary_file_name = self.summary_file_name
        if record_file_name is None:
            record_file_name = f'{self.solver_name}-{self.run_id}-{start_run_time}.csv'
        summary_info = self.recorder.summary_records(self.recorder.memory)
        end_run_time = time.time()
        clock_running_time = end_run_time - self.start_run_time
        run_info_dict = {
            'solver_name': self.solver_name,
            'seed': self.seed,
            'p_net_dataset_dir': self.p_net_dataset_dir,
            'v_nets_dataset_dir': self.v_nets_dataset_dir,
            'run_id': self.run_id,
            'start_run_time': start_run_time, 
            'clock_running_time': clock_running_time
        }
        for k, v in extra_summary_info.items():
            run_info_dict[k] = v
        info = {**summary_info, **run_info_dict}

        if self.if_save_records:
            record_path = self.recorder.save_records(record_file_name)
            summary_path = self.recorder.save_summary(info, summary_file_name)
                
        if self.verbose >= 1:
            pprint(info)
            if self.if_save_records:
                print(f'save records to {record_path}')
                print(f'save summary to {summary_path}')
        return info


class SolutionStepEnvironment(Environment):
    
    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, **kwargs):
        super(SolutionStepEnvironment, self).__init__(p_net, v_net_simulator, controller, recorder, counter, **kwargs)

    def step(self, solution):
        # Enter 
        self.solution = solution
        if self.solution['violation'] > 0: solution['result'] = False
        self.counter.count_solution(self.v_net, self.solution)
        # Success
        if solution['result']:
            assert len(solution['node_slots']) == self.v_net.num_nodes
            assert len(solution['link_paths']) == self.v_net.num_links
            self.solution['description'] = 'Success'
            total_p_resource_2 = self.counter.calculate_sum_network_resource(self.p_net)
            self.controller.deploy(self.v_net, self.p_net, self.solution)
            total_p_resource_1_n = self.counter.calculate_sum_node_resource(self.p_net)
            total_p_resource_1_e = self.counter.calculate_sum_link_resource(self.p_net)
            total_p_resource_0_n = self.counter.calculate_sum_node_resource(self.p_net_backup)
            total_p_resource_0_e = self.counter.calculate_sum_link_resource(self.p_net_backup)
            total_p_resource_1 = self.counter.calculate_sum_network_resource(self.p_net)
            total_p_resource_0 = self.counter.calculate_sum_network_resource(self.p_net_backup)
            # if total_p_resource_0 - total_p_resource_1 != solution['v_net_cost']:
            #     print('Unsafely Update!')
            assert total_p_resource_2 == total_p_resource_0
            assert total_p_resource_0_n - total_p_resource_1_n == solution['v_net_node_cost'], f"{total_p_resource_0_n - total_p_resource_1_n}, {solution['v_net_node_cost']}"
            assert total_p_resource_0_e - total_p_resource_1_e == solution['v_net_link_cost'], f"{total_p_resource_0_e - total_p_resource_1_e}, {solution['v_net_link_cost']}"
            assert total_p_resource_0 - total_p_resource_1 == solution['v_net_cost'], f"{total_p_resource_0 - total_p_resource_1}, {solution['v_net_cost']}"
        # Failure
        else:
            failure_reason = self.get_failure_reason(self.solution)
            self.rollback_for_failure(reason=failure_reason)
        record = self.count_and_add_record()
        done = self.transit_obs()
        return self.get_observation(), self.compute_reward(), done, self.get_info(record)

    def get_info(self, record={}):
        info = copy.deepcopy(record)
        return info

    def get_observation(self):
        return {'v_net': copy.deepcopy(self.v_net), 'p_net': copy.deepcopy(self.p_net)}

    def compute_reward(self):
        return 0

    def generate_action_mask(self):
        return np.arange(self.p_net.num_nodes)


class JointPRStepEnvironment(Environment):

    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, **kwargs):
        super(JointPRStepEnvironment, self).__init__(p_net, v_net_simulator, controller, recorder, counter, **kwargs)

    def step(self, p_node_id):
        place_and_route_result, place_and_route_info = self.controller.place_and_route(self.v_net, 
                                                                                        self.p_net, 
                                                                                        self.curr_v_node_id, 
                                                                                        p_node_id, 
                                                                                        self.solution, 
                                                                                        shortest_method=self.shortest_method, 
                                                                                        k=self.k_shortest)
        # Step Failure
        if not place_and_route_result:
            failure_reason = self.get_failure_reason(self.solution)
            self.rollback_for_failure(failure_reason)
        # Step Success
        else:
            self.curr_v_node_id += 1
            # VN Success ?
            if self.curr_v_node_id == self.v_net.num_nodes:
                self.solution['result'] = True
            else:
                record = self.solution.to_dict()
                return self.get_observation(), self.compute_reward(), False, self.get_info(record)

        record = self.count_and_add_record(self.v_net, self.p_net, self.solution)

        # obs transition
        if not place_and_route_result or self.solution['result']:
            done = self.transit_obs()
            if done:
                summary_info = self.summary_records()
                pprint(summary_info)
        else:
            done = False
        return self.get_observation(), self.compute_reward(), done, self.get_info(record)

    def get_observation(self):
        return {'v_node': self.v_net[self.curr_v_node_id], 'p_net': self.p_net}

    def compute_reward(self):
        return 0


if __name__ == '__main__':
    pass
