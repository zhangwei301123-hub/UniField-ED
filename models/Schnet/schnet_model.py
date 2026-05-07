import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_add
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


import src.schnetpack.nn as snn
import src.schnetpack.properties as properties
from src.schnetpack.representation import SchNet

class ED5SchNetModel(nn.Module):

    def __init__(self, config, output_dim=1):
        super().__init__()
        self.cutoff = config.get('cutoff', 10.0)
        self.max_neighbors = config.get('max_neighbors', 32)
        hidden_channels = config.get('hidden_channels', 128)


        radial_basis = snn.GaussianRBF(
            n_rbf=config.get('n_rbf', 300), 
            cutoff=self.cutoff
        )
        cutoff_fn = snn.CosineCutoff(cutoff=self.cutoff)
        
        self.model = SchNet(
            n_atom_basis=hidden_channels,
            n_interactions=config.get('n_interactions', 5),
            radial_basis=radial_basis,
            cutoff_fn=cutoff_fn,
            n_filters=hidden_channels,
        )
        

        self.out_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            snn.ShiftedSoftplus(),
            nn.Linear(hidden_channels // 2, output_dim)
        )

    def forward(self, input_dict):

        batch = input_dict['graph'] if 'graph' in input_dict else input_dict
        

        edge_index = radius_graph(
            batch.pos, 
            r=self.cutoff, 
            batch=batch.batch, 
            max_num_neighbors=self.max_neighbors
        )
        
        idx_j, idx_i = edge_index[0], edge_index[1]
        r_ij = batch.pos[idx_j] - batch.pos[idx_i]

        schnet_inputs = {
            properties.Z: batch.z,
            properties.Rij: r_ij,
            properties.idx_i: idx_i,
            properties.idx_j: idx_j,
        }

       
        results = self.model(schnet_inputs)
        atom_feats = results["scalar_representation"]
        
        mol_feats = scatter_add(atom_feats, batch.batch, dim=0)
        
        out = self.out_net(mol_feats)
        return out