import argparse

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
    return v.lower() in ('true', '1')

### Data ###
data_arg = add_argument_group('Data')

data_arg.add_argument('--node_attrs', type=list, default=['cpu', 'ram', 'rom'], help='Node attributes of PN or SFC')
data_arg.add_argument('--edge_attrs', type=list, default=['bw'], help='Edge attributes of PN or SFC')

# sfc simulator
data_arg.add_argument('--num_sfcs', type=int, default=2000, help='Number of SFCs')
data_arg.add_argument('--min_length', type=int, default=2, help='Minimum length of SFCs')
data_arg.add_argument('--max_length', type=int, default=15, help='Maximum length of SFCs')
data_arg.add_argument('--min_node_request', type=int, default=2, help='Minimum value of resource request of SFC nodes')
data_arg.add_argument('--max_node_request', type=int, default=30, help='Maximum value of resource request of SFC nodes')
data_arg.add_argument('--min_edge_request', type=int, default=2, help='Minimum value of resource request of SFC edges')
data_arg.add_argument('--max_edge_request', type=int, default=30, help='Maximum value of resource request of SFC edges')
data_arg.add_argument('--aver_lifetime', type=int, default=500, help='Average lifetime of SFCs')
data_arg.add_argument('--aver_arrival_rate', type=int, default=30, help='Average arrival rate of SFCs')

# physical network
data_arg.add_argument('--pn_num_nodes', type=int, default=100, help='Number of physical nodes')
data_arg.add_argument('--wm_alpha', type=float, default=0.2)
data_arg.add_argument('--wm_beta', type=float, default=0.5)
data_arg.add_argument('--min_node_capacity', type=int, default=50)
data_arg.add_argument('--max_node_capacity', type=int, default=100)
data_arg.add_argument('--min_edge_capacity', type=int, default=50)
data_arg.add_argument('--max_edge_capacity', type=int, default=100)

### Environment ###
env_arg = add_argument_group('Environment')
env_arg.add_argument('---pn_node_dataset', type=str, default='dataset/pn/nodes_data.csv', help='Path of PN node dataset')
env_arg.add_argument('---pn_edge_dataset', type=str, default='dataset/pn/edges_data.csv', help='Path of PN edge dataset')
env_arg.add_argument('---sfcs_dataset', type=str, default='dataset/sfc/sfcs_data.csv', help='Path of SFCs dataset')
env_arg.add_argument('---events_dataset', type=str, default='dataset/sfc/events_data.csv', help='Path of events dataset')

### Deployment ###
deployment_arg = add_argument_group('Deployment')
deployment_arg.add_argument('---reused_vnf', type=bool, default=False, help='Whether or not to allow to deploy several SFC nodes on the same VNF')

### Neural Network ###
net_arg = add_argument_group('Model')
net_arg.add_argument('---drl_gamma', type=float, default=0.95, help='Cumulative reward discount rate')
net_arg.add_argument('--explore_rate', type=float, default=0.9, help='Epsilon-greedy explore rate')
net_arg.add_argument('--actor_lr', type=float, default=0.00025, help='Actor learning rate')
net_arg.add_argument('--critic_lr', type=float, default=0.0005, help='Critic learning rate')
net_arg.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
net_arg.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension')
net_arg.add_argument('--num_layers', type=int, default=1, help='Number of GRU stacks\' layers')
net_arg.add_argument('--enc_units', type=int, default=64, help='Units of encoder GRU')
net_arg.add_argument('--dec_units', type=int, default=64, help='Units of decoder GRU')
net_arg.add_argument('--gnn_units', type=int, default=64, help='Units of decoder GNN')
net_arg.add_argument('---dropout_rate', type=float, default=0.2, help='Droput rate')
net_arg.add_argument('---l2reg_rate', type=float, default=2.5e-4, help='L2 regularization rate')

### Training ###
train_arg = add_argument_group('Training')
train_arg.add_argument('--run_mode', type=str, default='test', help='Number of epochs')
train_arg.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
data_arg.add_argument('--batch_size', type=int, default=64, help='Batch size of training')
train_arg.add_argument('--random_seed', type=int, default=1024, help='Random seed')
train_arg.add_argument('--save_model', type=str2bool, default=False, help='Save model')
train_arg.add_argument('--load_model', type=str2bool, default=False, help='Load model')
train_arg.add_argument('--save_dir', type=str, default='algo/save', help='Saver sub directory')
train_arg.add_argument('--log_dir', type=str, default='algo/summary', help='Summary writer log directory')
train_arg.add_argument('--load_from', type=str, default='algo/save', help='Loader sub directory')
#train_arg.add_argument('--lr_decay_step', type=int, default=5000, help='Lr1 decay step')
#train_arg.add_argument('--lr_decay_rate', type=float, default=0.96, help='Lr1 decay rate')

### Misc ###
misc_arg = add_argument_group('User options')
misc_arg.add_argument('--records_dir', type=str, default='records', help='Save records')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

if __name__ == "__main__":
    pass