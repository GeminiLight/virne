import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralTensorNetwork(torch.nn.Module):
    """
    Tensor Network module to calculate similarity vector.
    """
    def __init__(self, input_features, tensor_neurons):
        """
        :param args: Arguments object.
        """
        super(NeuralTensorNetwork, self).__init__()
        self.input_features = input_features
        self.tensor_neurons = tensor_neurons
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.input_features, self.input_features, self.tensor_neurons))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 2*self.input_features))
        self.bias = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.orthogonal_(self.weight_matrix)
        torch.nn.init.orthogonal_(self.weight_matrix_block)
        torch.nn.init.orthogonal_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size = len(embedding_1)
        scoring = torch.matmul(embedding_1, self.weight_matrix.view(self.input_features, -1))
        scoring = scoring.view(batch_size, self.input_features, -1).permute([0, 2, 1])
        scoring = torch.matmul(scoring, embedding_2.view(batch_size, self.input_features, 1)).view(batch_size, -1)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        block_scoring = torch.t(torch.mm(self.weight_matrix_block, torch.t(combined_representation)))
        scores = torch.relu(scoring + block_scoring + self.bias.view(-1))
        return scores
