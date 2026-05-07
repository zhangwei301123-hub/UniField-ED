import torch
import torch.nn as nn
from torch_scatter import scatter
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from visnet.models.visnet_block import ViSNetBlock 
from visnet.models.output_modules import EquivariantScalar 

class ViSNetModel(nn.Module):

    def __init__(self, config, output_dim=1):
        super().__init__()
        

        hidden_channels = config.get('hidden_channels', 128)
        self.hidden_channels = hidden_channels
        

        self.visnet_rep = ViSNetBlock(
            lmax=config.get('lmax', 2),
            vecnorm_type=config.get('vecnorm_type', 'none'),
            trainable_vecnorm=config.get('trainable_vecnorm', False),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 6),
            hidden_channels=hidden_channels, 
            num_rbf=config.get('num_rbf', 32),
            rbf_type=config.get('rbf_type', 'expnorm'),
            trainable_rbf=config.get('trainable_rbf', False),
            activation=config.get('activation', 'silu'),
            attn_activation=config.get('attn_activation', 'silu'),
            max_z=config.get('max_z', 100),
            cutoff=config.get('cutoff', 5.0),
            max_num_neighbors=config.get('max_num_neighbors', 32),
            vertex_type=config.get('vertex_type', 'Edge')
        )
        

        self.visnet_out = EquivariantScalar(
            hidden_channels=hidden_channels,
            output_dim=output_dim
        )
        
    def forward(self, input_dict):

        batch = input_dict['graph'] if 'graph' in input_dict else input_dict
        

        x, vec = self.visnet_rep(batch)
        
        pred_atom = self.visnet_out.pre_reduce(x, vec, batch.z, batch.pos, batch.batch)
        

        out = scatter(pred_atom, batch.batch, dim=0, reduce='sum')

        out = self.visnet_out.post_reduce(out)
        
        return out