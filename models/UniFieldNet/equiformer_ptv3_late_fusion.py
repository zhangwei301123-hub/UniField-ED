import os
import sys
import copy
import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_mean


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from PTv3.model import PointTransformerV3
    from nets.graph_attention_transformer import GraphAttentionTransformer 
except ImportError as e:
    print(f"❌ 导入失败！请仔细检查路径。详细错误: {e}")
    raise e


class EquiformerPTv3LateFusion(nn.Module):
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
        

        self.atom_proj = nn.Sequential(
            nn.Linear(self.equiformer_dim, self.equiformer_dim // 2),
            nn.SiLU(),
            nn.Linear(self.equiformer_dim // 2, equiformer_out_dim)
        )


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
        self.ptv3_dim = ptv3_args['dec_channels'][0] # Usually 64


        self.num_tasks = output_dim
        

        self.fusion_mlp = nn.Sequential(
            nn.Linear(equiformer_out_dim + self.ptv3_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, self.num_tasks) 
        )

    def forward(self, input_dict):
        atom_batch = input_dict['graph']
        ptv3_dict = input_dict['point_cloud']


        coord = ptv3_dict["coord"]
        grid_size = ptv3_dict.get("grid_size", 0.1)
        ptv3_dict["grid_coord"] = torch.div(coord - coord.min(0)[0], grid_size, rounding_mode='trunc').int()


        x_atom = self.equiformer(
            f_in=None, pos=atom_batch.pos, batch=atom_batch.batch, 
            node_atom=atom_batch.z, return_node_feats=True
        )
        pred_atom = self.atom_proj(x_atom) 
        

        equi_global_feat = scatter_mean(pred_atom, atom_batch.batch, dim=0)


        ptv3_out = self.ptv3(ptv3_dict)
        batch_cloud = self.offset2batch(ptv3_out.offset)
        

        ptv3_global_feat = scatter_mean(ptv3_out.feat, batch_cloud, dim=0)



        combined_feat = torch.cat([equi_global_feat, ptv3_global_feat], dim=1)
        

        out = self.fusion_mlp(combined_feat)

        return out

    def offset2batch(self, offset):
        counts = offset.clone()
        counts[1:] = offset[1:] - offset[:-1]
        batch_idx = torch.arange(len(offset), device=offset.device)
        return torch.repeat_interleave(batch_idx, counts)