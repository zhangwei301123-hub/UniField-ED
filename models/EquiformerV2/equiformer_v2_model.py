import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


from nets.equiformer_v2.equiformer_v2 import EquiformerV2

class ED5EquiformerV2Model(nn.Module):

    def __init__(self, config, output_dim=1):
        super().__init__()
        

        self.backbone = EquiformerV2(
            num_targets=output_dim, 
            use_pbc=config.get('use_pbc', False),
            otf_graph=config.get('otf_graph', True),
            max_neighbors=config.get('max_neighbors', 20),
            max_radius=config.get('max_radius', 12.0),
            max_num_elements=config.get('max_num_elements', 90),
            num_layers=config.get('num_layers', 4),
            sphere_channels=config.get('sphere_channels', 64),
            attn_hidden_channels=config.get('attn_hidden_channels', 64),
            num_heads=config.get('num_heads', 8),
            attn_alpha_channels=config.get('attn_alpha_channels', 64),
            attn_value_channels=config.get('attn_value_channels', 16),
            ffn_hidden_channels=config.get('ffn_hidden_channels', 128),
            norm_type=config.get('norm_type', 'layer_norm_sh'),
            lmax_list=config.get('lmax_list', [4]),
            mmax_list=config.get('mmax_list', [2]),
            grid_resolution=config.get('grid_resolution', 18),
            num_sphere_samples=config.get('num_sphere_samples', 128),
            edge_channels=config.get('edge_channels', 128),
            use_atom_edge_embedding=config.get('use_atom_edge_embedding', True),
            share_atom_edge_embedding=config.get('share_atom_edge_embedding', False),
            distance_function=config.get('distance_function', 'gaussian'),
            num_distance_basis=config.get('num_distance_basis', 512),
            attn_activation=config.get('attn_activation', 'silu'),
            use_s2_act_attn=config.get('use_s2_act_attn', False),
            use_attn_renorm=config.get('use_attn_renorm', True),
            ffn_activation=config.get('ffn_activation', 'silu'),
            use_gate_act=config.get('use_gate_act', False),
            use_grid_mlp=config.get('use_grid_mlp', True),
            use_sep_s2_act=config.get('use_sep_s2_act', True),
            alpha_drop=config.get('alpha_drop', 0.1),
            drop_path_rate=config.get('drop_path_rate', 0.05),
            proj_drop=config.get('proj_drop', 0.0),
            weight_init=config.get('weight_init', 'uniform')
        )


        self.backbone.regress_forces = False
        
    def forward(self, input_dict):

        batch = input_dict['graph'] if 'graph' in input_dict else input_dict
        

        if not hasattr(batch, 'atomic_numbers'):
            batch.atomic_numbers = batch.z
            

        if not hasattr(batch, 'natoms'):
            batch.natoms = torch.bincount(batch.batch).to(batch.batch.device)
        # =========================================================


        node_feats = self.backbone(batch) 
        

        if node_feats.ndim == 3:
            node_feats_scalar = node_feats[:, 0, :]  # [N, Channels]
        else:
            node_feats_scalar = node_feats
            
        mol_feats = global_add_pool(node_feats_scalar, batch.batch)
        
        return mol_feats