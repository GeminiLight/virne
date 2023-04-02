# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    
    def __init__(self, feature_dim, action_dim, embedding_dim=64):
        super(Actor, self).__init__()
        self.encoder = Encoder(feature_dim, action_dim, embedding_dim)
        self.decoder = Decoder(feature_dim, action_dim, embedding_dim)
        self._last_hidden_state = None
        self._last_cell_state = None

    def encode(self, x):
        obs = x['obs']
        outputs, (hidden_state, cell_state) = self.encoder(obs)
        self._last_hidden_state = hidden_state
        self._last_cell_state = cell_state
        return outputs

    def act(self, x):
        return self.decode(x)

    def decode(self, x):
        p_node_id = x['p_node_id']
        hidden_state = x['hidden_state']
        cell_state = x['cell_state']
        rnn_state = (hidden_state, cell_state)
        logits, output, (hidden_state, cell_state) = self.decoder(p_node_id, rnn_state)
        self._last_hidden_state = hidden_state
        self._last_cell_state = cell_state
        return logits

    def get_last_rnn_state(self):
        return self._last_hidden_state, self._last_cell_state

    def set_last_rnn_hidden(self, hidden_state):
        self._last_hidden_state = hidden_state
        self._last_cell_state = hidden_state


class Encoder(nn.Module):

    def __init__(self, feature_dim, action_dim, embedding_dim=64):
        super(Encoder, self).__init__()
        self.emb = nn.Linear(feature_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, embedding_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        embeddings = F.relu(self.emb(x))
        outputs, hidden_state = self.rnn(embeddings)
        return outputs, hidden_state


class Decoder(nn.Module):
    
    def __init__(self, feature_dim, action_dim, embedding_dim=64):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(action_dim+1, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, embedding_dim)
        self.lin = nn.Linear(embedding_dim, action_dim)

    def forward(self, x, rnn_state):
        x = x.unsqueeze(0)
        embedding = F.relu(self.emb(x))
        output, rnn_state = self.rnn(embedding, rnn_state)
        logits = self.lin(output)
        return logits, output, rnn_state
