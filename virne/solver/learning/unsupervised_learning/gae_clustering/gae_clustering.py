import copy
import torch
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score

from virne.core import Solution
from virne.solver.learning.utils import get_available_device, load_pyg_data_from_network
from virne.solver.base_solver import Solver, SolverRegistry
from .model import Encoder, Discriminator, GraphVineDecoder, ARGVA


@SolverRegistry.register(solver_name='gae_clustering', solver_type='u_learning')
class GaeClusteringSolver(Solver):
    """
    A unsupervised learning solver that uses Graph Auto-Encoder (GAE) to cluster the physical nodes.
    
    References:
        - Farzad Habibi et al. "Accelerating Virtual Network Embedding with Graph Neural Networks". In CNSM, 2020.
    
    Attributes:
        num_features (int): The number of features of the physical nodes.
        num_clusters (int): The number of clusters.
        argva (ARGVA): The auto-encoder model.
        discriminator_optimizer (torch.optim): The optimizer for the discriminator.
        argva_optimizer (torch.optim): The optimizer for the auto-encoder.
        pretrain_max_epochs (int): The maximum number of epochs for pre-training.
        fine_tune_max_epochs (int): The maximum number of epochs for fine-tuning.
        max_num_attempts (int): The maximum number of attempts to find a good clustering.
        num_arrived_v_nets (int): The number of virtual networks that have arrived.
    """
    def __init__(self, controller, recorder, counter, logger, num_features=1, num_clusters=4, **kwargs):
        super(GaeClusteringSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        self.num_features = num_features
        self.num_clusters = num_clusters
        embedding_dim = 64
        encoder = Encoder(self.num_features, embedding_dim, embedding_dim)
        discriminator = Discriminator(embedding_dim, embedding_dim * 2, embedding_dim)
        decoder = GraphVineDecoder(embedding_dim, embedding_dim, int(embedding_dim / 2), self.num_features)

        self.argva = ARGVA(encoder, discriminator, decoder=decoder)
        self.discriminator_optimizer = torch.optim.Adam(self.argva.discriminator.parameters(), lr=0.001)
        self.argva_optimizer = torch.optim.Adam(self.argva.parameters(), lr=5*0.001)
        self.device = get_available_device()

        self.pretrain_max_epochs = 100
        self.fine_tune_max_epochs = 5
        self.max_num_attempts = 5
        self.pertrained = False
        self.num_arrived_v_nets = 0

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, dict): config = vars(config)
        config = copy.deepcopy(config)
        num_features = config.pop('num_v_net_node_attrs', 1)
        num_clusters = config.pop('num_clusters', 4)
        controller = None
        recorder = None
        counter = None
        return cls(num_features, num_clusters,  controller, recorder, counter, logger, **config)

    def cluster(self, network):
        def load_data(network):
            network_data = load_pyg_data_from_network(network, normailize_nodes_data=True, normailize_method='standardize')
            network_data.edge_attr = network_data.edge_attr / network_data.edge_attr.sum()
            data = network_data.clone()
            self.argva.to(self.device)
            data = data.to(self.device)
            return data

        def train(data):
            self.argva.train()
            self.argva_optimizer.zero_grad()
            z = self.argva.encode(data.x, data.edge_index, data.edge_attr)

            for i in range(self.fine_tune_max_epochs):
                self.argva.discriminator.train()
                self.discriminator_optimizer.zero_grad()
                discriminator_loss = self.argva.discriminator_loss(z)
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

            loss = self.argva.recon_loss(z, data.edge_index, data.x)
            loss = loss + (1 / data.num_nodes) * self.argva.kl_loss()
            loss.backward()
            self.argva_optimizer.step()
            return loss

        def test(data):
            self.argva.eval()
            kmeans_input = []
            with torch.no_grad():
                z = self.argva.encode(data.x, data.edge_index, data.edge_attr)
                kmeans_input = z.cpu().data.numpy()
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(kmeans_input)
            pred_label = kmeans.predict(kmeans_input)
            X = data.x.cpu().data

            s = silhouette_score(X, pred_label)
            davies = davies_bouldin_score(X, pred_label)
            return s, davies

        max_epoch = self.fine_tune_max_epochs if self.pertrained else self.pretrain_max_epochs
        data = load_data(network)
        self.pertrained = True

        # train and test
        for epoch in range(max_epoch):
            loss = train(data)
            if self.verbose or epoch == max_epoch-1:
                s, davies = test(data)
                if self.verbose >= 2: print('Epoch: {:05d}, '
                    'Train Loss: {:.3f}, Silhoeute: {:.3f}, Davies: {:.3f}'.format(epoch, loss, s, davies))
        # cluster
        with torch.no_grad():
            z = self.argva.encode(data.x, data.edge_index, data.edge_attr)
        self.clusters_index = KMeans(n_clusters=self.num_clusters).fit_predict(z.cpu().data)
        return self.clusters_index

    def select_action(self, obs, mask=None):
        return self.solve(obs['v_net'], obs['p_net'])

    def solve(self, instance) -> Solution:
        v_net, p_net = instance['v_net'], instance['p_net']
        self.num_arrived_v_nets += 1
        if not (self.num_arrived_v_nets - 1) % 50: self.cluster(p_net)

        feasible_solutions = []
        for cluster_id in range(self.num_clusters):
            remain_num_attempts = self.max_num_attempts
            while remain_num_attempts != 0:
                # Try to embed request in some adjacent nodes from different clusters
                temp_p_net = copy.deepcopy(p_net)

                alpha = 1.6
                max_depth = 3
                starting_node = self.sample_from_p_net(self.clusters_index, cluster_id)
                max_visit_nodes = v_net.num_nodes * alpha
                solution = self.controller.bfs_deploy(v_net, temp_p_net, list(v_net.nodes), starting_node, max_visit=max_visit_nodes, max_depth=max_depth)
                remain_num_attempts -= 1

                if solution['result']:
                    feasible_solutions.append((solution, temp_p_net))
                    self.counter.count_solution(v_net, solution)

        if len(feasible_solutions) != 0:
            solution, temp_p_net = sorted(feasible_solutions, key=lambda t: t[0]['v_net_cost'])[0]
            p_net.__dict__ = copy.deepcopy(temp_p_net.__dict__)
            return solution
        else:
            return Solution.from_v_net(v_net)

    def sample_from_p_net(self, cluster_index, cluster_id):
        """return a node from one cluste"""
        cluster_index = np.where(cluster_index == cluster_id)[0]
        sample_node = np.random.choice(cluster_index, 1, replace=False)[0]
        return sample_node

        # feasible_solutions = []
        # for cluster_id in range(self.num_clusters):
        #     remain_num_attempts = max_num_attempts
        #     attempt_deploy_result = False
        #     temp_p_net = copy.deepcopy(p_net)
        #     while not attempt_deploy_result and remain_num_attempts != 0:
        #         starting_node = self.sample_from_p_net(temp_p_net, self.clusters_index, cluster_id)
        #         solution, deployed_temp_p_net = self.try_deploy_in_center(starting_node, temp_p_net, v_net)
        #         remain_num_attempts -= 1

        #     if attempt_deploy_result:
        #         feasible_solutions.append((solution, deployed_temp_p_net))

        # if len(feasible_solutions) != 0:
        #     solution, temp_p_net = sorted(feasible_solutions, key=lambda t: t[0]['v_net_cost'])[0]
        #     return True, solution, temp_p_net
        # else:
        #     return False, None, p_net
