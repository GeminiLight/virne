# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import os
import tqdm
import pprint

from .controller import Controller
from .recorder import Recorder
from .counter import Counter
from .solution import Solution
from config import show_config, save_config
from data.physical_network import PhysicalNetwork
from data.virtual_network_request_simulator import VirtualNetworkRequestSimulator
from utils import get_p_net_dataset_dir_from_setting


class Scenario:

    def __init__(self, env, solver, config):
        self.env = env
        self.solver = solver
        self.config = config
        self.verbose = config.verbose

    @classmethod
    def from_config(cls, Env, Solver, config):
        # Create basic class: controller, recorder, counter, recorder
        counter = Counter(config.v_sim_setting['node_attrs_setting'], 
                          config.v_sim_setting['link_attrs_setting'], 
                          **vars(config))
        controller = Controller(config.v_sim_setting['node_attrs_setting'], 
                                config.v_sim_setting['link_attrs_setting'], 
                                **vars(config))
        recorder = Recorder(counter, **vars(config))

        # Create p_net and v_net simulator
        config.p_net_dataset_dir = get_p_net_dataset_dir_from_setting(config.p_net_setting)
        print(config.p_net_dataset_dir)
        if os.path.exists(config.p_net_dataset_dir):
            p_net = PhysicalNetwork.load_dataset(config.p_net_dataset_dir)
            print(f'Load Physical Network from {config.p_net_dataset_dir}') if config.verbose >= 1 else None
        else:
            p_net = PhysicalNetwork.from_setting(config.p_net_setting)
            print(f'*** Generate Physical Network from setting')
        v_net_simulator = VirtualNetworkRequestSimulator.from_setting(config.v_sim_setting)
        print(f'Create VNR Simulator from setting') if config.verbose >= 1 else None

        # create env and solver
        env = Env(p_net, v_net_simulator, controller, recorder, counter, **vars(config))
        solver = Solver(controller, recorder, counter, **vars(config))

        # Create scenario
        scenario = cls(env, solver, config)
        if config.verbose >= 2: show_config(config)
        if config.if_save_config: save_config(config)

        return scenario

    def reset(self):
        pass

    def ready(self):
        # load pretrained model
        if hasattr(self.solver, 'load_model') and self.config.pretrained_model_path not in ['None', '']:
            if os.path.exists(self.config.pretrained_model_path):
                self.solver.load_model(self.config.pretrained_model_path)
            else:
                print(f'Load pretrained model failed: Path does not exist {self.config.pretrained_model_path}')
        # execute pretrain
        if hasattr(self.solver, 'learn') and self.config.num_train_epochs > 0:
            print(f"\n{'-' * 20}  Pretrain  {'-' * 20}\n")
            self.solver.learn(self.env, num_epochs=self.config.num_train_epochs)
            print(f"\n{'-' * 20}    Done    {'-' * 20}\n")
        # set eval mode
        if hasattr(self.solver, 'eval'):
            self.solver.eval()

class BasicScenario(Scenario):

    def __init__(self, env, solver, config):
        super(BasicScenario, self).__init__(env, solver, config)

    def run(self):
        self.ready()

        for epoch_id in range(self.config.start_epoch, self.config.start_epoch + self.config.num_epochs):
            print(f'\nEpoch {epoch_id}') if self.verbose >= 2 else None
            instance = self.env.reset()

            pbar = tqdm.tqdm(desc=f'Running with {self.config.solver_name} in epoch {epoch_id}', total=self.env.num_v_nets) if self.verbose <= 1 else None

            while True:
                solution = self.solver.solve(instance)

                next_instance, _, done, info = self.env.step(solution)

                if pbar is not None: 
                    pbar.update(1)
                    pbar.set_postfix({
                        'ac': f'{info["success_count"] / info["v_net_count"]:1.2f}',
                        'r2c': f'{info["total_r2c"]:1.2f}',
                        'inservice': f'{info["inservice_count"]:05d}',
                    })

                if done:
                    break
                instance = next_instance
  
            if pbar is not None: pbar.close()
            summary_info = self.env.summary_records()
            if self.verbose == 0:
                pprint.pprint(summary_info)


class TimeWindowScenario(Scenario):

    def __init__(self, env, solver, config):
        super(TimeWindowScenario, self).__init__(env, solver, config)
        self.time_window_size = config.get('time_window_size', 100)

    def reset(self):
        self.current_time_window = 0
        self.next_event_id = 0
        return super().reset()

    def _receive(self):
        next_time_window = self.current_time_window + self.time_window_size
        enter_event_list = []
        leave_event_list = []
        while self.next_event_id < len(self.v_net_simulator.events) and self.v_net_simulator.events[self.next_event_id]['time'] <= next_time_window:
            if self.v_net_simulator.events[self.next_event_id]['type'] == 1:
                enter_event_list.append(self.v_net_simulator.events[self.next_event_id])
            else:
                leave_event_list.append(self.v_net_simulator.events[self.next_event_id])
            self.next_event_id += 1
        return enter_event_list, leave_event_list

    def _transit(self, solution_dict):
        return NotImplementedError

    def run(self):
        self.ready()
        
        for epoch_id in range(self.config.start_epoch, self.config.start_epoch + self.config.num_epochs):
            print(f'\nEpoch {epoch_id}') if self.verbose >= 2 else None
            pbar = tqdm.tqdm(desc=f'Running with {self.solver.name} in epoch {epoch_id}', total=self.env.num_v_nets) if self.verbose <= 1 else None
            instance = self.env.reset()

            current_event_id = 0
            events_list = self.env.v_net_simulator.events
            for current_time in range(0, int(events_list[-1]['time'] + self.time_window_size + 1), self.time_window_size):
                enter_event_list = []
                while events_list[current_event_id]['time'] < current_time:
                    # enter
                    if events_list[current_event_id]['type'] == 1:
                        enter_event_list.append(events_list[current_event_id])
                    # leave
                    else:
                        v_net_id = events_list[current_event_id]['v_net_id']
                        solution = Solution(self.v_net_simulator.v_nets[v_net_id])
                        solution = self.recorder.get_record(v_net_id=v_net_id)
                        self.controller.release(self.v_net_simulator.v_nets[v_net_id], self.p_net, solution)
                        self.solution['description'] = 'Leave Event'
                        record = self.count_and_add_record()
                    current_event_id += 1

                for enter_event in  enter_event_list:
                    solution = self.solver.solve(instance)
                    next_instance, _, done, info = self.env.step(solution)

                    if pbar is not None: 
                        pbar.update(1)
                        pbar.set_postfix({
                            'ac': f'{info["success_count"] / info["v_net_count"]:1.2f}',
                            'r2c': f'{info["total_r2c"]:1.2f}',
                            'inservice': f'{info["inservice_count"]:05d}',
                        })

                    if done:
                        break
                    instance = next_instance
  
            if pbar is not None: pbar.close()
            summary_info = self.env.summary_records()
            if self.verbose == 0:
                pprint.pprint(summary_info)

