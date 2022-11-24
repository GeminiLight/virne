import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    
    def __init__(self, feature_dim, action_dim, embedding_dim=64):
        super(Actor, self).__init__()
        self.encoder = Encoder(feature_dim, action_dim, embedding_dim)
        self.decoder = Decoder(feature_dim, action_dim, embedding_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x, hidden):
        return self.decoder(x, hidden)


class Encoder(nn.Module):

    def __init__(self, feature_dim, action_dim, embedding_dim=64):
        super(Encoder, self).__init__()
        self.emb = nn.Linear(feature_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, embedding_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        embeddings = F.relu(self.emb(x))
        outputs, hidden = self.rnn(embeddings)
        return outputs, hidden


class Decoder(nn.Module):
    
    def __init__(self, feature_dim, action_dim, embedding_dim=64):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(action_dim+1, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, embedding_dim)
        self.lin = nn.Linear(embedding_dim, action_dim)

    def forward(self, x, hidden):
        x = x.unsqueeze(0)
        embedding = F.relu(self.emb(x))
        output, hidden = self.rnn(embedding, hidden)
        logits = self.lin(output)
        return logits, hidden
