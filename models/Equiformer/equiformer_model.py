import os
import sys
import copy
import torch
import torch.nn as nn
from torch_scatter import scatter

# ==================动态挂载路径 ==================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from nets.graph_attention_transformer import GraphAttentionTransformer 
except ImportError as e:
    print(f"❌ 导入失败！请仔细检查路径。详细错误: {e}")
    raise e


class ED5Equiformer(nn.Module):
    def __init__(self, config, output_dim=7, normalizer=None):
        super().__init__()
        
        # 挂载 mean 和 std
        if normalizer is not None:
            self.register_buffer('mean', normalizer['mean'])
            self.register_buffer('std', normalizer['std'])
        else:
            self.register_buffer('mean', torch.zeros(output_dim))
            self.register_buffer('std', torch.ones(output_dim))

        equiformer_args = copy.deepcopy(config.get('equiformer_args', {}))
        

        equiformer_out_dim = equiformer_args.pop('output_dim', 64)


        self.equiformer = GraphAttentionTransformer(**equiformer_args)
        
        irreps_feature_str = equiformer_args.get('irreps_feature', '512x0e')
        self.equiformer_dim = int(irreps_feature_str.split('x')[0])

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


        x_atom = self.equiformer(
            f_in=None, pos=atom_batch.pos, batch=atom_batch.batch, 
            node_atom=atom_batch.z, return_node_feats=True
        )
        

        pred_atom = self.atom_proj(x_atom) 
        

        out_sum = self.equiformer.scale_scatter(pred_atom, atom_batch.batch, dim=0)
        if self.equiformer.scale is not None:
            out_sum = self.equiformer.scale * out_sum
            
        out_mean = scatter(pred_atom, atom_batch.batch, dim=0, reduce='mean')
        out_max = scatter(pred_atom, atom_batch.batch, dim=0, reduce='max')
        out = torch.cat([out_sum, out_mean, out_max], dim=-1)
        

        task_outputs = [head(out) for head in self.independent_heads]
        out = torch.cat(task_outputs, dim=1) 
        
 
        return out