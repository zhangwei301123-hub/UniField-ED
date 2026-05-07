import os
import sys
import copy
import torch
import torch.nn as nn
from torch_cluster import radius
from torch_scatter import scatter
from torch_geometric.utils import softmax as pyg_softmax


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from PTv3.model import PointTransformerV3
    from nets.graph_attention_transformer import GraphAttentionTransformer 
except ImportError as e:
    print(f"❌ 导入失败！请仔细检查路径。详细错误: {e}")
    raise e



class CloudToAtomInteraction(nn.Module):
    def __init__(self, atom_dim, cloud_dim, interaction_radius=4.0, num_heads=4):
        super().__init__()
        self.atom_dim = atom_dim
        self.cloud_dim = cloud_dim
        self.interaction_radius = interaction_radius
        
        self.q_atom = nn.Linear(atom_dim, atom_dim)
        self.k_cloud = nn.Linear(cloud_dim, atom_dim)
        self.v_cloud = nn.Linear(cloud_dim, atom_dim)
        

        
        self.out_proj = nn.Linear(atom_dim, atom_dim)
        self.norm = nn.LayerNorm(atom_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_atom, pos_atom, batch_atom, x_cloud, pos_cloud, batch_cloud):
        edge_index = radius(x=pos_cloud, y=pos_atom, 
                            r=self.interaction_radius,
                            batch_x=batch_cloud, batch_y=batch_atom,
                            max_num_neighbors=256) 
        
        idx_atom, idx_cloud = edge_index[0], edge_index[1]
        
        Q, K, V = self.q_atom(x_atom), self.k_cloud(x_cloud), self.v_cloud(x_cloud)
        q_vec, k_vec, v_vec = Q[idx_atom], K[idx_cloud], V[idx_cloud]
        

        scores = (q_vec * k_vec).sum(dim=-1) / (self.atom_dim ** 0.5)

        attn_weights = pyg_softmax(scores, idx_atom)
        msg = v_vec * attn_weights.unsqueeze(-1)
        aggr_feat = scatter(msg, idx_atom, dim=0, dim_size=x_atom.size(0), reduce='sum')
        x_atom_new = self.norm(x_atom + self.dropout(self.out_proj(self.act(aggr_feat))))
        
        return x_atom_new


class UniFieldNet_NoDist(nn.Module):
    def __init__(self, config, output_dim=7, normalizer=None):
        super().__init__()
        
        if normalizer is not None:
            self.register_buffer('mean', normalizer['mean'])
            self.register_buffer('std', normalizer['std'])
        else:
            self.register_buffer('mean', torch.zeros(output_dim))
            self.register_buffer('std', torch.ones(output_dim))

        equiformer_args = copy.deepcopy(config.get('equiformer_args', {}))
        ptv3_args = copy.deepcopy(config.get('ptv3_args', {}))
        
        equiformer_out_dim = equiformer_args.pop('output_dim', 64) 

        self.equiformer = GraphAttentionTransformer(**equiformer_args)
        
        irreps_feature_str = equiformer_args.get('irreps_feature', '512x0e')
        self.equiformer_dim = int(irreps_feature_str.split('x')[0])
        
        enc_depths = ptv3_args.get('enc_depths', (2, 2, 2, 6, 2))
        num_stages = len(enc_depths)
        patch_size_base = ptv3_args.get('patch_size', 256)
        
        stride = ptv3_args.get('stride', (2,) * (num_stages - 1))
        dec_depths = ptv3_args.get('dec_depths', (2,) * (num_stages - 1))
        enc_patch_size = (patch_size_base,) * num_stages
        dec_patch_size = (patch_size_base,) * (num_stages - 1)
        
        enc_num_head = ptv3_args.get('enc_num_head', [2**(i+1) for i in range(num_stages)])
        dec_num_head = ptv3_args.get('dec_num_head', [4 * (2**i) for i in range(num_stages - 1)])

        self.ptv3 = PointTransformerV3(
            in_channels=ptv3_args.get('in_channels', 1),
            enc_depths=enc_depths, enc_channels=ptv3_args.get('enc_channels'),
            dec_channels=ptv3_args.get('dec_channels'), stride=stride, dec_depths=dec_depths,
            enc_patch_size=enc_patch_size, dec_patch_size=dec_patch_size,
            enc_num_head=enc_num_head, dec_num_head=dec_num_head,
            mlp_ratio=4, qkv_bias=True, enable_flash=True, enable_rpe=False,
            pdnorm_ln=False, cls_mode=False 
        )
        
        self.ptv3_dim = ptv3_args['dec_channels'][0]
        self.interaction = CloudToAtomInteraction(
            atom_dim=self.equiformer_dim, cloud_dim=self.ptv3_dim, interaction_radius=4.0
        )

        self.num_tasks = output_dim 
        self.head_input_dim = config.get('head_input_dim', 64) 
        
        self.atom_proj = nn.Sequential(
            nn.Linear(self.equiformer_dim, self.equiformer_dim // 2),
            nn.SiLU(),
            nn.Linear(self.equiformer_dim // 2, self.head_input_dim)
        )
        
        self.independent_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_input_dim * 3, 64), 
                nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1)
            ) for _ in range(self.num_tasks)
        ])

    def forward(self, input_dict):
        atom_batch = input_dict['graph']
        ptv3_dict = input_dict['point_cloud']

        coord = ptv3_dict["coord"]
        grid_size = ptv3_dict.get("grid_size", 0.1)
        grid_coord = torch.div(
            coord - coord.min(0)[0], grid_size, rounding_mode='trunc'
        ).int()
        ptv3_dict["grid_coord"] = grid_coord

        x_atom = self.equiformer(
            f_in=None, pos=atom_batch.pos, batch=atom_batch.batch, 
            node_atom=atom_batch.z, return_node_feats=True
        )
        
        ptv3_out = self.ptv3(ptv3_dict)
        x_cloud = ptv3_out.feat
        pos_cloud = ptv3_out.coord
        batch_cloud = self.offset2batch(ptv3_out.offset)
        
        x_atom_refined = self.interaction(
            x_atom=x_atom, pos_atom=atom_batch.pos, batch_atom=atom_batch.batch,
            x_cloud=x_cloud, pos_cloud=pos_cloud, batch_cloud=batch_cloud
        )
        
        pred_atom = self.atom_proj(x_atom_refined) 
        
        out_sum = self.equiformer.scale_scatter(pred_atom, atom_batch.batch, dim=0)
        if self.equiformer.scale is not None:
            out_sum = self.equiformer.scale * out_sum
            
        out_mean = scatter(pred_atom, atom_batch.batch, dim=0, reduce='mean')
        out_max = scatter(pred_atom, atom_batch.batch, dim=0, reduce='max')
        out = torch.cat([out_sum, out_mean, out_max], dim=-1)
        
        task_outputs = [head(out) for head in self.independent_heads]
        out = torch.cat(task_outputs, dim=1) 
        
        return out

    def offset2batch(self, offset):
        counts = offset.clone()
        counts[1:] = offset[1:] - offset[:-1]
        batch_idx = torch.arange(len(offset), device=offset.device)
        return torch.repeat_interleave(batch_idx, counts)