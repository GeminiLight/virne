import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PositionalEncoder(nn.Module):

    def __init__(self, embedding_dim, dropout_prob=1.0, max_len=50, method='add'):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob < 1. else nn.Identity()
        # Compute the positional encodingss
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.method = method
        
    def forward(self, x):
        pe_embeddings = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        pe_embeddings = pe_embeddings.repeat(x.shape[0], 1, 1)
        # if len(x.shape) == 3:
            # pe_embeddings = pe_embeddings.unsqueeze(0).repeat(x.shape[0], 1, 1)
        if self.method == 'add':
            x = x + pe_embeddings
        elif self.method == 'concat':
            x = torch.concat([x, pe_embeddings], dim=-1)
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):

    def __init__(
            self,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            num_heads=1,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.n_heads = num_heads

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // self.n_heads
        if key_dim is None:
            key_dim = val_dim
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(self.n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(self.n_heads, input_dim, key_dim))
        # self.W_val = nn.Parameter(torch.Tensor(self.n_heads, input_dim, key_dim))
        
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, attn_mask=None):
        """

        :param q: queries (batch_size, num_query, input_dim)
        :param exchange: (batch_size, 2)
        :return:
        """
        # compute self-attention
        if k is None:
            k = q
        if v is None:
            v = k

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = k.size()
        num_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        kflat = k.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shape_k = (self.n_heads, batch_size, graph_size, -1)
        shape_q = (self.n_heads, batch_size, num_query, -1)

        # Calculate queries
        Q = torch.matmul(qflat, self.W_query).view(shape_q)  # (n_heads, num_query, graph_size, key/val_size)
        # Calculate keys and values
        K = torch.matmul(kflat, self.W_key).view(shape_k)    # (n_heads, batch_size, graph_size, key/val_size)
        # V = torch.matmul(kflat, self.W_val).view(shape_k)

        # Calculate compatibility 
        compatibility_raw = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # (n_heads, batch_size, num_query, problem_size)
        compatibility = torch.tanh(compatibility_raw.mean(dim=0)) * 10.            # (batch_size, num_query, problem_size)

        # if attn_mask is not None:
        #     compatibility.masked_fill_(attn_mask, float('-inf'))

        # attn_weights = F.softmax(compatibility, dim=-1)
        # output = torch.matmul(attn_weights, V)  # (n_heads, batch_size, num_query, key_dim)
        # output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, num_query, -1)  # (batch_size, num_query, val_dim)
        return compatibility


class MultiHeadAttentionWithOutput(nn.Module):
    def __init__(self, embedding_dim, num_heads=1, d_k=None, d_v=None, dropout=0.1):
        super(MultiHeadAttentionWithOutput, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k if d_k is not None else embedding_dim
        self.d_v = d_v if d_v is not None else embedding_dim
        
        self.W_q = nn.Linear(embedding_dim, num_heads*self.d_k, bias=False)
        self.W_k = nn.Linear(embedding_dim, num_heads*self.d_k, bias=False)
        self.W_v = nn.Linear(embedding_dim, num_heads*self.d_v, bias=False)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_heads*self.d_v, embedding_dim)
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1,2)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.W_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1,2)    # (batch_size, num_heads, seq_len_k, d_k)
        V = self.W_v(value).view(batch_size, seq_len_v, self.num_heads, self.d_v).transpose(1,2)  # (batch_size, num_heads, seq_len_v, d_v)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        attn_weights = self.softmax(scores)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len_q, d_v)
        
        # Concatenation of heads
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, seq_len_q, self.num_heads*self.d_v)  # (batch_size, seq_len_q, num_heads*d_v)
        
        # Final linear layer
        output = self.fc(attn_output)  # (batch_size, seq_len_q, embedding_dim)
        return output, attn_weights
