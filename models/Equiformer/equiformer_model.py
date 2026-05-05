import os
import sys
import copy
import torch
import torch.nn as nn
from torch_scatter import scatter

# ================== 💡 动态挂载路径 ==================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from nets.graph_attention_transformer import GraphAttentionTransformer 
except ImportError as e:
    print(f"❌ 导入失败！请仔细检查路径。详细错误: {e}")
    raise e

# ================== 主模型包装 (纯 Equiformer 消融版) ==================

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
        
        # 弹出 output_dim 防止底层报错
        equiformer_out_dim = equiformer_args.pop('output_dim', 64)

        # 1. 仅初始化 Equiformer
        self.equiformer = GraphAttentionTransformer(**equiformer_args)
        
        irreps_feature_str = equiformer_args.get('irreps_feature', '512x0e')
        self.equiformer_dim = int(irreps_feature_str.split('x')[0])

        self.num_tasks = output_dim 
        self.head_input_dim = config.get('head_input_dim', 64) 
        
        # 2. 自定义原子特征投影层 (将 Equiformer 特征平滑降维)
        self.atom_proj = nn.Sequential(
            nn.Linear(self.equiformer_dim, self.equiformer_dim // 2),
            nn.SiLU(),
            nn.Linear(self.equiformer_dim // 2, self.head_input_dim)
        )
        
        # 3. 独立多任务头
        self.independent_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_input_dim * 3, 64), 
                nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1)
            ) for _ in range(self.num_tasks)
        ])

    def forward(self, input_dict):
        # 1. 仅提取原子图数据
        atom_batch = input_dict['graph']

        # 2. Equiformer 提取图特征
        x_atom = self.equiformer(
            f_in=None, pos=atom_batch.pos, batch=atom_batch.batch, 
            node_atom=atom_batch.z, return_node_feats=True
        )
        
        # 3. 投影降维
        pred_atom = self.atom_proj(x_atom) 
        
        # 4. Sum + Mean + Max 混合池化 (保留你引以为傲的池化设计)
        out_sum = self.equiformer.scale_scatter(pred_atom, atom_batch.batch, dim=0)
        if self.equiformer.scale is not None:
            out_sum = self.equiformer.scale * out_sum
            
        out_mean = scatter(pred_atom, atom_batch.batch, dim=0, reduce='mean')
        out_max = scatter(pred_atom, atom_batch.batch, dim=0, reduce='max')
        out = torch.cat([out_sum, out_mean, out_max], dim=-1)
        
        # 5. 分发给多任务头
        task_outputs = [head(out) for head in self.independent_heads]
        out = torch.cat(task_outputs, dim=1) 
        
        # 返回归一化空间的输出
        return out