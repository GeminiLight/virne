import torch
import numpy as np
from torch_geometric.data import Data, Batch

from args import get_config
from virne.solver.learning.hrl_ppo_degat.hrl_ppo_degat_env import HRLPPODEGATEnv
from virne.solver.learning.hrl_ppo_degat.model_old import LowerNet, UpperNet
from virne.solver.learning.utils import get_pyg_data, get_pyg_batch


def preprocess_one_obs(obs):
    p_net_data = get_pyg_data(obs['p_net_x'], obs['p_net_edge_index'], obs['p_net_edge_attr'])
    v_net_data = get_pyg_data(obs['v_net_x'], obs['v_net_edge_index'], obs['v_net_edge_attr'])
    p_net_data = Batch.from_data_list([p_net_data])
    v_net_data = Batch.from_data_list([v_net_data])
    v_net_attrs = torch.tensor([obs['v_net_attrs']])
    upper_obs = {
        'p_net_data': p_net_data, 
        'v_net_data': v_net_data, 
        'v_net_attrs': v_net_attrs
    }
    return upper_obs

def preprocess_batch_obs(obs_batch):
    batch_size = len(obs_batch)
    batch_p_net_x = [obs_batch[i]['p_net_x'] for i in range(batch_size)]
    batch_p_net_edge_index = [obs_batch[i]['p_net_edge_index'] for i in range(batch_size)]
    batch_p_net_edge_attr = [obs_batch[i]['p_net_edge_attr'] for i in range(batch_size)]
    p_net_batch = get_pyg_batch(batch_p_net_x, batch_p_net_edge_index, batch_p_net_edge_attr)
    
    batch_v_net_x = [obs_batch[i]['v_net_x'] for i in range(batch_size)]
    batch_v_net_edge_index = [obs_batch[i]['v_net_edge_index'] for i in range(batch_size)]
    batch_v_net_edge_attr = [obs_batch[i]['v_net_edge_attr'] for i in range(batch_size)]
    v_net_batch = get_pyg_batch(batch_v_net_x, batch_v_net_edge_index, batch_v_net_edge_attr)

    v_net_attrs_batch = np.array([obs_batch[i]['v_net_attrs'] for i in range(batch_size)])
    v_net_attrs_batch = torch.tensor(v_net_attrs_batch)

    upper_obs_batch = {
        'p_net_data': p_net_batch, 
        'v_net_data': v_net_batch, 
        'v_net_attrs': v_net_attrs_batch
    }
    return upper_obs_batch


config = get_config()
env = HRLPPODEGATEnv.from_config(config)

env.ready(0)
obs = env.get_observation()

embedding_dim = 128
hidden_dim = 128


upper_net = UpperNet(p_net_node_dim=7, p_net_edge_dim=2, v_net_node_dim=3, v_net_edge_dim=1, v_net_attrs_dim=2, output_dim=2)
lower_net = LowerNet(p_net_node_dim=7, p_net_edge_dim=2, v_net_node_dim=1, v_net_edge_dim=1, v_net_attrs_dim=2, output_dim=100)

upper_hidden = torch.zeros(1, hidden_dim)
obs_tensor = preprocess_one_obs(obs)
upper_out, upper_hidden = upper_net(obs_tensor['p_net_data'], obs_tensor['v_net_data'], upper_hidden, obs_tensor['v_net_attrs'])

observations = [obs, obs, obs, obs]
upper_hidden_batch = torch.zeros(4, hidden_dim)
obs_batch = preprocess_batch_obs(observations)
upper_out_batch, upper_hidden_batch = upper_net(obs_batch['p_net_data'], obs_batch['v_net_data'], upper_hidden_batch, obs_batch['v_net_attrs'])


# lower agents
def preprocess_one_obs_lower(obs):
    p_net_data = get_pyg_data(obs['p_net_x'], obs['p_net_edge_index'], obs['p_net_edge_attr'])
    p_net_data = Batch.from_data_list([p_net_data])
    v_net_attrs = torch.tensor([obs['v_net_attrs']])
    lower_obs = {
        'p_net_data': p_net_data, 
        'v_net_attrs': v_net_attrs
    }
    return lower_obs


lower_init_obs = obs
lower_init_obs_tensor = obs_tensor
v_node_embedings = lower_net.encoder(lower_init_obs_tensor['v_net_data'])  # required_grad


num_v_node = len(lower_init_obs['v_net_x'])
lower_hidden = torch.zeros(1, hidden_dim)
for i in range(num_v_node):
    index = torch.LongTensor([i]).unsqueeze(0).unsqueeze(-1)
    v_node_embeding = v_node_embedings.gather(1, index.expand(v_node_embedings.size()[0], -1, v_node_embedings.size()[-1])).squeeze(1)
    lower_out, lower_hidden = lower_net.decoder(lower_init_obs_tensor['p_net_data'], v_node_embeding, lower_hidden, lower_init_obs_tensor['v_net_attrs'])
    
    lower_obs = {
        'p_net_data': p_net_data, 
        'v_node_embeding': v_node_embeding, 
        'hidden': lower_hidden, 
        'v_net_attrs': v_net_attrs
    }