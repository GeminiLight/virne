import torch
import numpy as np
from torch_geometric.data import Data, Batch

from config import get_config
from solver.learning.hrl_ppo_degat.hrl_ppo_degat_env import HRLPPODEGATEnv
from solver.learning.hrl_ppo_degat.model_old import LowerNet, UpperNet
from solver.learning.utils import get_pyg_data, get_pyg_batch


def preprocess_one_obs(obs):
    pn_data = get_pyg_data(obs['pn_x'], obs['pn_edge_index'], obs['pn_edge_attr'])
    vn_data = get_pyg_data(obs['vn_x'], obs['vn_edge_index'], obs['vn_edge_attr'])
    pn_data = Batch.from_data_list([pn_data])
    vn_data = Batch.from_data_list([vn_data])
    vn_attrs = torch.tensor([obs['vn_attrs']])
    upper_obs = {
        'pn_data': pn_data, 
        'vn_data': vn_data, 
        'vn_attrs': vn_attrs
    }
    return upper_obs

def preprocess_batch_obs(obs_batch):
    batch_size = len(obs_batch)
    batch_pn_x = [obs_batch[i]['pn_x'] for i in range(batch_size)]
    batch_pn_edge_index = [obs_batch[i]['pn_edge_index'] for i in range(batch_size)]
    batch_pn_edge_attr = [obs_batch[i]['pn_edge_attr'] for i in range(batch_size)]
    pn_batch = get_pyg_batch(batch_pn_x, batch_pn_edge_index, batch_pn_edge_attr)
    
    batch_vn_x = [obs_batch[i]['vn_x'] for i in range(batch_size)]
    batch_vn_edge_index = [obs_batch[i]['vn_edge_index'] for i in range(batch_size)]
    batch_vn_edge_attr = [obs_batch[i]['vn_edge_attr'] for i in range(batch_size)]
    vn_batch = get_pyg_batch(batch_vn_x, batch_vn_edge_index, batch_vn_edge_attr)

    vn_attrs_batch = np.array([obs_batch[i]['vn_attrs'] for i in range(batch_size)])
    vn_attrs_batch = torch.tensor(vn_attrs_batch)

    upper_obs_batch = {
        'pn_data': pn_batch, 
        'vn_data': vn_batch, 
        'vn_attrs': vn_attrs_batch
    }
    return upper_obs_batch


config = get_config()
env = HRLPPODEGATEnv.from_config(config)

env.ready(0)
obs = env.get_observation()

embedding_dim = 128
hidden_dim = 128


upper_net = UpperNet(pn_node_dim=7, pn_edge_dim=2, vn_node_dim=3, vn_edge_dim=1, vn_attrs_dim=2, output_dim=2)
lower_net = LowerNet(pn_node_dim=7, pn_edge_dim=2, vn_node_dim=1, vn_edge_dim=1, vn_attrs_dim=2, output_dim=100)

upper_hidden = torch.zeros(1, hidden_dim)
obs_tensor = preprocess_one_obs(obs)
upper_out, upper_hidden = upper_net(obs_tensor['pn_data'], obs_tensor['vn_data'], upper_hidden, obs_tensor['vn_attrs'])

observations = [obs, obs, obs, obs]
upper_hidden_batch = torch.zeros(4, hidden_dim)
obs_batch = preprocess_batch_obs(observations)
upper_out_batch, upper_hidden_batch = upper_net(obs_batch['pn_data'], obs_batch['vn_data'], upper_hidden_batch, obs_batch['vn_attrs'])


# lower agents
def preprocess_one_obs_lower(obs):
    pn_data = get_pyg_data(obs['pn_x'], obs['pn_edge_index'], obs['pn_edge_attr'])
    pn_data = Batch.from_data_list([pn_data])
    vn_attrs = torch.tensor([obs['vn_attrs']])
    lower_obs = {
        'pn_data': pn_data, 
        'vn_attrs': vn_attrs
    }
    return lower_obs


lower_init_obs = obs
lower_init_obs_tensor = obs_tensor
vnf_embedings = lower_net.encoder(lower_init_obs_tensor['vn_data'])  # required_grad


num_vnf = len(lower_init_obs['vn_x'])
lower_hidden = torch.zeros(1, hidden_dim)
for i in range(num_vnf):
    index = torch.LongTensor([i]).unsqueeze(0).unsqueeze(-1)
    vnf_embeding = vnf_embedings.gather(1, index.expand(vnf_embedings.size()[0], -1, vnf_embedings.size()[-1])).squeeze(1)
    lower_out, lower_hidden = lower_net.decoder(lower_init_obs_tensor['pn_data'], vnf_embeding, lower_hidden, lower_init_obs_tensor['vn_attrs'])
    
    lower_obs = {
        'pn_data': pn_data, 
        'vnf_embeding': vnf_embeding, 
        'hidden': lower_hidden, 
        'vn_attrs': vn_attrs
    }