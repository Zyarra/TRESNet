import torch.nn as nn
from project.layers import LayerNormalize, Residual


class MLP_Block(nn.Module):
    def __init__(self, in_features, hidden_features=None, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Allows the model to jointly attend to information
        from different representation subspaces.
        See reference: Attention Is All You Need
    """
    def __init__(self, dim, num_heads=8, attn_dropout=0.1, dropout=0.1):
        """
        dim: the input and dimension of the model.
        heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.nn1 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.nn1(x)
        x = self.dropout(x)
        return x


class Transformer(nn.Module):
    """
    dim: the input and output dimension of the transformer model
    heads: the number of heads in the multi-head-attention models.
    depth: the number of encoder and decoder layers
    mlp_dim: the dimension of the feedforward network model
    dropout: the dropout value.
    """
    def __init__(self, dim, num_heads, depth, mlp_hidden_dim, dropout=01., attn_dropout=01.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(
                    dim, num_heads=num_heads, attn_dropout=attn_dropout, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(
                    in_features=dim, hidden_features=mlp_hidden_dim, dropout=dropout)))
            ]))

    def forward(self, x):
        for attention, mlp in self.layers:
            x = attention(x)
            x = mlp(x)
        return x
