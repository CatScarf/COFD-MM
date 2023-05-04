import os
from typing import List, Tuple, Union
import logging

import torch
from torch import nn, Tensor
from torch.nn.utils import rnn
from torch.nn import functional as F

import config

real_dir = os.path.dirname(__file__)

def find_idx(a: Tensor, b: Tensor, miss: int=-1):
    """Find the first index of b in a, return array like a."""
    a, b = a.clone(), b.clone()
    invalid = ~torch.isin(a, b)
    a[invalid] = b[0]
    sorter = torch.argsort(b)
    b_to_a: Tensor = sorter[torch.searchsorted(b, a, sorter=sorter)]
    b_to_a[invalid] = miss
    return b_to_a

def norm(x: Tensor, dim: int = 0) -> Tensor:
    x = x - x.mean(dim=dim, keepdim=True)
    x = x / x.std(dim=dim, keepdim=True)
    return x

def stat_missing(x: Tensor):
    a = int((x == -1).sum())
    b = int(x.numel())
    return f'{a}/{b}({a / b * 100:.2f}%)'

class Encoder(nn.Module):
    def __init__(self, ndim: int, edim: int, embdim: int, mlp_layers: List[int], args: config.Args):
        super().__init__()

        self.args = args

        # RNN Encoder.
        if args.encoder == 'lstm':
            self.encoder = nn.LSTM(edim, embdim)
            self.encode = lambda x : self.encoder.forward(x)[1][0][-1].squeeze(0)
            mlp_layers = [ndim + embdim, embdim] + mlp_layers
        else:
            raise ValueError(f'Unknown encoder: {args.encoder}')

        # MLP Classifier.
        mlp_layers_inout = list(zip(mlp_layers[:-1], mlp_layers[1:]))
        self.mlp_layers = nn.ModuleList([nn.Linear(in_, out_) for in_, out_ in mlp_layers_inout])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n) for n in mlp_layers[1:]]) if args.norm else None

    def forward(self, nfeat: Tensor, efeat: rnn.PackedSequence) -> Tuple[Tensor, Tensor]:
        # RNN Encoder.
        x = self.encode(efeat)
        x = torch.cat([x, nfeat], dim=1)
        
        # MLP Classifier.
        emb = None
        for i, layer in enumerate(self.mlp_layers):
            if i == 1:
                emb = norm(x)
            x = layer(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if i < len(self.mlp_layers) - 1:
                x = F.relu(x)
        y = F.softmax(x, dim=1)
        assert emb is not None
        return emb, y

    def reset_parameters(self):
        for layer in self.mlp_layers:
            layer.reset_parameters()  # type: ignore
        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()  # type: ignore
        self.encoder.reset_parameters()
