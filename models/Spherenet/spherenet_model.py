import torch
import torch.nn as nn
import os
import sys
import numpy as np
import math
if not hasattr(np, 'math'):
    np.math = math



current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


from dig.threedgraph.method import SphereNet

class ED5SphereNetModel(nn.Module):

    def __init__(self, config, output_dim=1):
        super().__init__()
        

        self.model = SphereNet(
            energy_and_force=False,
            cutoff=config.get('cutoff', 5.0),
            num_layers=config.get('num_layers', 4),
            hidden_channels=config.get('hidden_channels', 128),
            out_channels=output_dim,  
            int_emb_size=64,
            basis_emb_size_dist=8,
            basis_emb_size_angle=8,
            basis_emb_size_torsion=8,
            out_emb_channels=256,
            num_spherical=config.get('num_spherical', 3),
            num_radial=config.get('num_radial', 6),
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3
        )

    def forward(self, input_dict):
    
        batch = input_dict['graph'] if 'graph' in input_dict else input_dict
        
        
        out = self.model(batch)
        
        return out