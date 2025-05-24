# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import numpy as np
from gym import spaces
from typing import Any, Dict, Tuple, List, Union, Optional, Type, Callable


import torch
import numpy as np


from virne.network import PhysicalNetwork, VirtualNetwork
from virne.core import Controller, Recorder, Counter, Solution, Logger

from virne.solver import SolverRegistry
from virne.solver.learning.rl_policy import CnnActorCritic
from virne.solver.learning.rl_core import JointPRStepInstanceRLEnv, PlaceStepInstanceRLEnv
from virne.solver.learning.rl_core.rl_solver import PGSolver, A2CSolver, PPOSolver, A3CSolver
from virne.solver.learning.rl_core.instance_agent import InstanceAgent
from virne.solver.learning.utils import get_pyg_data
from torch_geometric.data import Data, Batch, HeteroData


class TensorConvertor:
    """
    Utility class for converting various observation formats to PyTorch tensors for RL policies.
    All methods are static for easy use and extension.
    """
    @staticmethod
    def p_net_obs_as_tensor(obs, device):
        if isinstance(obs, dict):
            assert 'p_net_x' in obs, "Observation dictionary must contain 'p_net_x' key."
            assert 'p_net_edge_index' in obs, "Observation dictionary must contain 'p_net_edge_index' key."
            obs_p_net_edge_attr = obs.get('p_net_edge_attr', None)
            tensor_obs_p_net = get_pyg_data(obs['p_net_x'], obs['p_net_edge_index'], obs_p_net_edge_attr)
            tensor_obs_p_net = Batch.from_data_list([tensor_obs_p_net]).to(device)
            return {'p_net': tensor_obs_p_net}
        elif isinstance(obs, list):
            p_net_data_list = []
            for observation in obs:
                assert 'p_net_x' in observation, "Observation dictionary must contain 'p_net_x' key."
                assert 'p_net_edge_index' in observation, "Observation dictionary must contain 'p_net_edge_index' key."
                obs_p_net_edge_attr = observation.get('p_net_edge_attr', None)
                p_net_data = get_pyg_data(observation['p_net_x'], observation['p_net_edge_index'], obs_p_net_edge_attr)
                p_net_data_list.append(p_net_data)
            tensor_obs_p_net = Batch.from_data_list(p_net_data_list).to(device)
            return {'p_net': tensor_obs_p_net}
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")

    @staticmethod
    def p_net_x_obs_as_tensor(obs, device):
        if isinstance(obs, dict):
            tensor_obs_p_net_x = torch.FloatTensor(np.array([obs['p_net_x']])).to(device)
            return {'p_net_x': tensor_obs_p_net_x}
        elif isinstance(obs, list):
            p_net_x_list = []
            for observation in obs:
                p_net_x_list.append(observation['p_net_x'])
            tensor_obs_p_net_x = torch.FloatTensor(np.array(p_net_x_list)).to(device)
            return {'p_net_x': tensor_obs_p_net_x}
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")

    @staticmethod
    def v_net_x_obs_as_tensor(obs, device):
        if isinstance(obs, dict):
            tensor_obs_v_net_x = torch.FloatTensor(np.array([obs['v_net_x']])).to(device)
            return {'v_net_x': tensor_obs_v_net_x}
        elif isinstance(obs, list):
            v_net_x_list = []
            for observation in obs:
                v_net_x_list.append(observation['v_net_x'])
            tensor_obs_v_net_x = torch.FloatTensor(np.array(v_net_x_list)).to(device)
            return {'v_net_x': tensor_obs_v_net_x}
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")

    @staticmethod
    def v_net_obs_as_tensor(obs, device):
        if isinstance(obs, dict):
            assert 'v_net_x' in obs, "Observation dictionary must contain 'v_net_x' key."
            assert 'v_net_edge_index' in obs, "Observation dictionary must contain 'v_net_edge_index' key."
            obs_v_net_edge_attr = obs.get('v_net_edge_attr', None)
            tensor_obs_v_net = get_pyg_data(obs['v_net_x'], obs['v_net_edge_index'], obs_v_net_edge_attr)
            tensor_obs_v_net = Batch.from_data_list([tensor_obs_v_net]).to(device)
            return {'v_net': tensor_obs_v_net}
        elif isinstance(obs, list):
            v_net_data_list = []
            for observation in obs:
                assert 'v_net_x' in observation, "Observation dictionary must contain 'v_net_x' key."
                assert 'v_net_edge_index' in observation, "Observation dictionary must contain 'v_net_edge_index' key."
                obs_v_net_edge_attr = observation.get('v_net_edge_attr', None)
                v_net_data = get_pyg_data(observation['v_net_x'], observation['v_net_edge_index'], obs_v_net_edge_attr)
                v_net_data_list.append(v_net_data)
            tensor_obs_v_net = Batch.from_data_list(v_net_data_list).to(device)
            return {'v_net': tensor_obs_v_net}
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")

    @staticmethod
    def v_node_obs_as_tensor(obs, device):
        if isinstance(obs, dict):
            tensor_obs_v_net_x = torch.FloatTensor(np.array([obs['v_node_x']])).to(device)
            return {'v_node_x': tensor_obs_v_net_x}
        elif isinstance(obs, list):
            v_node_x_list = []
            for observation in obs:
                v_node_x_list.append(observation['v_node_x'])
            tensor_obs_v_net_x = torch.FloatTensor(np.array(v_node_x_list)).to(device)
            return {'v_node_x': tensor_obs_v_net_x}
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")

    @staticmethod
    def rnn_state_obs_as_tensor(obs, device):
        rnn_state_obs_item_keys = set(['p_node_id', 'hidden_state', 'cell_state', 'encoder_outputs'])
        if isinstance(obs, dict):
            tensor_rnn_state_obs_dict = {}
            for item in rnn_state_obs_item_keys:
                if item not in obs:
                    continue
                if item == 'p_node_id':
                    tensor_rnn_state_obs_dict[item] = torch.LongTensor(np.array([obs[item]])).to(device)
                elif item == 'encoder_outputs':
                    tensor_rnn_state_obs_dict[item] = torch.FloatTensor(np.array([obs[item]])).to(device)
                else:
                    tensor_rnn_state_obs_dict[item] = torch.FloatTensor(np.array([obs[item]])).to(device)
            # tensor_rnn_state_obs_dict['mask'] = torch.ones_like(tensor_rnn_state_obs_dict['encoder_outputs'], dtype=torch.float).to(device)
            tensor_rnn_state_obs_dict['mask'] = None
            return tensor_rnn_state_obs_dict
        elif isinstance(obs, list):
            # batch
            rnn_state_obs_list_dict = {}
            for observation in obs:
                for item in rnn_state_obs_item_keys:
                    if item not in observation:
                        continue
                    if item not in rnn_state_obs_list_dict:
                        rnn_state_obs_list_dict[item] = []
                    rnn_state_obs_list_dict[item].append(observation[item])
            tensor_rnn_state_obs_dict = {}
            for item in rnn_state_obs_list_dict:
                if item == 'p_node_id':
                    tensor_rnn_state_obs_dict[item] = torch.LongTensor(np.array(rnn_state_obs_list_dict[item])).to(device)
                elif item == 'encoder_outputs':
                    sequences = rnn_state_obs_list_dict[item]
                    max_length = max([seq.shape[0] for seq in sequences])
                    padded_sequences = np.zeros((len(sequences), max_length, sequences[0].shape[1]))
                    mask = np.zeros((len(sequences), max_length), dtype='bool')
                    for i, seq in enumerate(sequences):
                        seq_len = seq.shape[0]
                        padded_sequences[i, :seq_len, :] = seq
                        mask[i, :seq_len] = 1
                    tensor_rnn_state_obs_dict['mask'] =  torch.FloatTensor(mask).to(device)
                    tensor_rnn_state_obs_dict[item] = torch.FloatTensor(np.array(padded_sequences)).to(device)
                else:
                    tensor_rnn_state_obs_dict[item] = torch.FloatTensor(np.array(rnn_state_obs_list_dict[item])).to(device)
            return tensor_rnn_state_obs_dict
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")

    @staticmethod
    def general_obs_as_tensor(obs, device):
        general_obs_item_keys = set(['curr_v_node_id', 'action_mask', 'v_net_size'])
        if isinstance(obs, dict):
            tensor_general_obs_dict = {}
            for item in general_obs_item_keys:
                if item not in obs:
                    continue
                if item == 'curr_v_node_id':
                    tensor_general_obs_dict[item] = torch.LongTensor(np.array([obs[item]])).to(device)
                elif item == 'v_net_size':
                    tensor_general_obs_dict[item] = torch.FloatTensor(np.array([obs[item]])).to(device)
                else:
                    tensor_general_obs_dict[item] = torch.FloatTensor(np.array([obs[item]])).to(device)
            return tensor_general_obs_dict
        elif isinstance(obs, list):
            general_obs_list_dict = {}
            for observation in obs:
                for item in general_obs_item_keys:
                    if item not in observation:
                        continue
                    if item not in general_obs_list_dict:
                        general_obs_list_dict[item] = []
                    general_obs_list_dict[item].append(observation[item])
            tensor_general_obs_dict = {}
            for item in general_obs_list_dict:
                if item == 'curr_v_node_id':
                    tensor_general_obs_dict[item] = torch.LongTensor(np.array(general_obs_list_dict[item])).to(device)
                elif item == 'v_net_size':
                    tensor_general_obs_dict[item] = torch.FloatTensor(np.array(general_obs_list_dict[item])).to(device)
                else:
                    tensor_general_obs_dict[item] = torch.FloatTensor(np.array(general_obs_list_dict[item])).to(device)
            return tensor_general_obs_dict
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")

    @staticmethod
    def array_obs_as_tensor(obs, device):
        if isinstance(obs, dict):
            tensor_obs = {}
            for obs_key, obs_value in obs.items():
                tensor_obs[obs_key] = torch.FloatTensor(np.array([obs_value])).to(device)
            return tensor_obs
        elif isinstance(obs, list):
            tensor_obs = {}
            obs_list = {}
            for observation in obs:
                for obs_key, obs_value in observation.items():
                    if obs_key not in obs_list:
                        obs_list[obs_key] = []
                    obs_list[obs_key].append(obs_value)
            for obs_key, obs_value in obs_list.items():
                tensor_obs[obs_key] = torch.FloatTensor(np.array(obs_value)).to(device)
            return tensor_obs
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")

    @staticmethod
    def obs_as_tensor_for_gnn_mlp(obs, device):
        tensor_p_net_obs = TensorConvertor.p_net_obs_as_tensor(obs, device)
        tensor_v_node_obs = TensorConvertor.v_node_obs_as_tensor(obs, device)
        tensor_general_obs = TensorConvertor.general_obs_as_tensor(obs, device)
        return {**tensor_p_net_obs, **tensor_v_node_obs, **tensor_general_obs}

    @staticmethod
    def obs_as_tensor_for_mlp(obs, device):
        tensor_p_net_obs = TensorConvertor.p_net_x_obs_as_tensor(obs, device) # (n, b)
        tensor_v_node_obs = TensorConvertor.v_node_obs_as_tensor(obs, device)  # (a, )
        tensor_general_obs = TensorConvertor.general_obs_as_tensor(obs, device)
        v_node_x = tensor_v_node_obs['v_node_x']
        try:
            v_node_x = v_node_x.unsqueeze(1).expand(-1, tensor_p_net_obs['p_net_x'].shape[1], -1)
            p_net_x = torch.cat((tensor_p_net_obs['p_net_x'], v_node_x), dim=-1)
        except Exception as e:
            import pdb; pdb.set_trace()
        tensor_p_net_x_obs = {'p_net_x': p_net_x}
        return {**tensor_p_net_x_obs, **tensor_general_obs}

    @staticmethod
    def obs_as_tensor_for_cnn(obs, device):
        mlp_obs = TensorConvertor.obs_as_tensor_for_mlp(obs, device)
        mlp_obs['p_net_x'] = mlp_obs['p_net_x'].unsqueeze(1)
        return mlp_obs

    @staticmethod
    def obs_as_tensor_for_mlp_fc(obs, device):
        tensor_p_nodes_obs = TensorConvertor.p_net_x_obs_as_tensor(obs, device)
        tensor_v_node_obs = TensorConvertor.v_node_obs_as_tensor(obs, device)
        tensor_general_obs = TensorConvertor.general_obs_as_tensor(obs, device)
        # flatten the p_nodes_x
        p_nodes_x = torch.flatten(tensor_p_nodes_obs['p_net_x'], start_dim=1)
        pv_nodes_x = torch.cat((p_nodes_x, tensor_v_node_obs['v_node_x']), dim=-1)
        pv_nodes_obs = {'pv_nodes_x': pv_nodes_x, }
        return {**pv_nodes_obs, **tensor_general_obs}

    @staticmethod
    def obs_as_tensor_for_dual_gnn(obs, device):
        tensor_p_net_obs = TensorConvertor.p_net_obs_as_tensor(obs, device)
        tensor_v_net_obs = TensorConvertor.v_net_obs_as_tensor(obs, device)
        tensor_general_obs = TensorConvertor.general_obs_as_tensor(obs, device)
        return {**tensor_p_net_obs, **tensor_v_net_obs, **tensor_general_obs}

    @staticmethod
    def obs_as_tensor_for_att(obs, device):
        tensor_obs_p_net_x = TensorConvertor.p_net_x_obs_as_tensor(obs, device)
        tensor_obs_v_node_x = TensorConvertor.v_node_obs_as_tensor(obs, device)
        tensor_general_obs = TensorConvertor.general_obs_as_tensor(obs, device)
        return {**tensor_obs_p_net_x, **tensor_obs_v_node_x, **tensor_general_obs}

    @staticmethod
    def obs_as_tensor_for_gnn_seq2seq(obs, device):
        # one
        if isinstance(obs, dict):
            """Preprocess the observation to adapte to batch mode."""
            tensor_p_net_obs = TensorConvertor.p_net_obs_as_tensor(obs, device)
            tensor_rnn_state_obs = TensorConvertor.rnn_state_obs_as_tensor(obs, device)
            tensor_general_obs = TensorConvertor.general_obs_as_tensor(obs, device)
            return {**tensor_p_net_obs, **tensor_rnn_state_obs, **tensor_general_obs}
        # batch
        elif isinstance(obs, list):
            tensor_p_net_obs = TensorConvertor.p_net_obs_as_tensor(obs, device)
            tensor_rnn_state_obs = TensorConvertor.rnn_state_obs_as_tensor(obs, device)
            tensor_general_obs = TensorConvertor.general_obs_as_tensor(obs, device)
            return {**tensor_p_net_obs, **tensor_rnn_state_obs, **tensor_general_obs}
        else:
            raise ValueError('obs type error')

    @staticmethod
    def obs_as_tensor_for_hetero_gnn(obs, device):
        # one
        if isinstance(obs, dict):
            """Preprocess the observation to adapt to batch mode."""
            observation = obs
            x_dict = {'p': observation['p_net_x'], 'v': observation['v_net_x']}
            edge_index_dict = {('p', 'connect', 'p'): observation['p_net_edge_index'], ('v', 'connect', 'v'): observation['v_net_edge_index'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_index'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_index']}
            edge_attr_dict = {('p', 'connect', 'p'): observation['p_net_edge_attr'], ('v', 'connect', 'v'): observation['v_net_edge_attr'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_attr'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_attr']}
            hetero_data = get_pyg_hetero_data(x_dict, edge_index_dict, edge_attr_dict)
            aug_edge_index_dict_a = {('p', 'connect', 'p'): observation['p_net_edge_index'], ('v', 'connect', 'v'): observation['v_net_aug_edge_index'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_index'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_index']}
            aug_edge_attr_dict_a = {('p', 'connect', 'p'): observation['p_net_edge_attr'], ('v', 'connect', 'v'): observation['v_net_aug_edge_attr'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_attr'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_attr']}
            aug_edge_index_dict_b = {('p', 'connect', 'p'): observation['p_net_aug_edge_index'], ('v', 'connect', 'v'): observation['v_net_edge_index'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_index'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_index']}
            aug_edge_attr_dict_b = {('p', 'connect', 'p'): observation['p_net_aug_edge_attr'], ('v', 'connect', 'v'): observation['v_net_edge_attr'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_attr'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_attr']}
            aug_hetero_data_a = get_pyg_hetero_data(x_dict, aug_edge_index_dict_a, aug_edge_attr_dict_a)
            aug_hetero_data_b = get_pyg_hetero_data(x_dict, aug_edge_index_dict_b, aug_edge_attr_dict_b)
            obs_aug_hetero_data_a = Batch.from_data_list([aug_hetero_data_a]).to(device)
            obs_aug_hetero_data_b = Batch.from_data_list([aug_hetero_data_b]).to(device)
            obs_hetero_data = Batch.from_data_list([hetero_data]).to(device)
            obs_curr_v_node_id = torch.LongTensor(np.array([observation['curr_v_node_id']])).to(device)
            obs_action_mask = torch.FloatTensor(np.array([observation['action_mask']])).to(device)
            obs_v_net_size = torch.LongTensor(np.array([observation['v_net_size']])).to(device)
            # return {'p_net': obs_p_net, 'v_net': obs_v_net, 'hetero_data': obs_hetero_data, 'curr_v_node_id': obs_curr_v_node_id, 'action_mask': obs_action_mask, 'v_net_size': obs_v_net_size}
            return {'hetero_data': obs_hetero_data, 'curr_v_node_id': obs_curr_v_node_id, 'action_mask': obs_action_mask, 'v_net_size': obs_v_net_size, 'aug_hetero_data_a': obs_aug_hetero_data_a, 'aug_hetero_data_b': obs_aug_hetero_data_b}
        # batch
        elif isinstance(obs, list):
            import pdb; pdb.set_trace()
            p_net_data_list, v_net_data_list, hetero_data_list, curr_v_node_id_list, action_mask_list, v_net_size_list = [], [], [], [], [], []
            aug_hetero_data_list_a, aug_hetero_data_list_b = [], []
            for observation in obs:
                x_dict = {'p': observation['p_net_x'], 'v': observation['v_net_x']}
                edge_index_dict = {('p', 'connect', 'p'): observation['p_net_edge_index'], ('v', 'connect', 'v'): observation['v_net_edge_index'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_index'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_index']}
                edge_attr_dict = {('p', 'connect', 'p'): observation['p_net_edge_attr'], ('v', 'connect', 'v'): observation['v_net_edge_attr'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_attr'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_attr']}
                hetero_data = get_pyg_hetero_data(x_dict, edge_index_dict, edge_attr_dict)
                hetero_data_list.append(hetero_data)
                aug_edge_index_dict_a = {('p', 'connect', 'p'): observation['p_net_edge_index'], ('v', 'connect', 'v'): observation['v_net_aug_edge_index'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_index'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_index']}
                aug_edge_attr_dict_a = {('p', 'connect', 'p'): observation['p_net_edge_attr'], ('v', 'connect', 'v'): observation['v_net_aug_edge_attr'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_attr'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_attr']}
                aug_edge_index_dict_b = {('p', 'connect', 'p'): observation['p_net_aug_edge_index'], ('v', 'connect', 'v'): observation['v_net_edge_index'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_index'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_index']}
                aug_edge_attr_dict_b = {('p', 'connect', 'p'): observation['p_net_aug_edge_attr'], ('v', 'connect', 'v'): observation['v_net_edge_attr'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_attr'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_attr']}
                aug_hetero_data_a = get_pyg_hetero_data(x_dict, aug_edge_index_dict_a, aug_edge_attr_dict_a)
                aug_hetero_data_b = get_pyg_hetero_data(x_dict, aug_edge_index_dict_b, aug_edge_attr_dict_b)
                aug_hetero_data_list_a.append(aug_hetero_data_a)
                aug_hetero_data_list_b.append(aug_hetero_data_b)
                curr_v_node_id_list.append(observation['curr_v_node_id'])
                action_mask_list.append(observation['action_mask'])
                v_net_size_list.append(observation['v_net_size'])
            obs_hetero_data = Batch.from_data_list(hetero_data_list).to(device)
            obs_aug_hetero_data_a = Batch.from_data_list(aug_hetero_data_list_a).to(device)
            obs_aug_hetero_data_b = Batch.from_data_list(aug_hetero_data_list_b).to(device)
            obs_curr_v_node_id = torch.LongTensor(np.array(curr_v_node_id_list)).to(device)
            obs_v_net_size = torch.FloatTensor(np.array(v_net_size_list)).to(device)
            # Get the length of the longest sequence
            max_len_action_mask = max(len(seq) for seq in action_mask_list)
            # Pad all sequences with zeros up to the max length
            padded_action_mask = np.zeros((len(action_mask_list), max_len_action_mask))
            for i, seq in enumerate(action_mask_list):
                padded_action_mask[i, :len(seq)] = seq
            obs_action_mask = torch.FloatTensor(np.array(padded_action_mask)).to(device)
            return {'hetero_data': obs_hetero_data, 'curr_v_node_id': obs_curr_v_node_id, 'action_mask': obs_action_mask, 'v_net_size': obs_v_net_size, 'aug_hetero_data_a': obs_aug_hetero_data_a, 'aug_hetero_data_b': obs_aug_hetero_data_b}
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")


def get_pyg_hetero_data(x_dict, edge_index_dict, edge_attr_dict, reverse_edge=True):
    """Preprocess the observation to adapt to batch mode."""
    hetero_data = HeteroData()
    for key, value in x_dict.items():
        hetero_data[key].x = torch.tensor(value, dtype=torch.float32)
    for key, value in edge_index_dict.items():
        hetero_data[key[0], key[1], key[2]].edge_index = torch.tensor(value).long()
    for key, value in edge_attr_dict.items():
        hetero_data[key[0], key[1], key[2]].edge_attr = torch.tensor(value, dtype=torch.float32)
    if reverse_edge:
        for key, value in edge_index_dict.items():
            if key[0] == key[2]: continue
            hetero_data[key[2], key[1], key[0]].edge_index = hetero_data[key[0], key[1], key[2]].edge_index.flip(0)
        for key, value in edge_attr_dict.items():
            if key[0] == key[2]: continue
            hetero_data[key[2], key[1], key[0]].edge_attr = hetero_data[key[0], key[1], key[2]].edge_attr.flip(0)
    return hetero_data
