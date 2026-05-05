import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
import os
import sys

# 💡 动态挂载当前路径，确保能找到 gotennet.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from gotennet import GotenNetWrapper
except ImportError as e:
    print(f"❌ 导入 GotenNetWrapper 失败，请确保 gotennet.py 在 {current_dir} 目录下。")
    raise e

# 💡 内部集成专属的 CosineCutoff
class CosineCutoff(nn.Module):
    def __init__(self, cutoff_val=5.0):
        super().__init__()
        self.cutoff = cutoff_val

    def forward(self, dist):
        mask = (dist <= self.cutoff).float()
        dist = torch.clamp(dist, max=self.cutoff) 
        cutoffs = 0.5 * (torch.cos(dist * torch.pi / self.cutoff) + 1.0)
        return mask * cutoffs

class ED5GotenNetModel(nn.Module):
    """
    [GotenNet 适配版]
    包含：Encoder (GotenNet) + Decoder (Atom-wise MLP) + Pooling (Sum)
    """
    def __init__(self, config, output_dim=1):
        super().__init__()
        
        cutoff_val = config.get('cutoff', 10.0)
        hidden_channels = config.get('hidden_channels', 256)
        num_layers = config.get('num_layers', 8)
        
        # 1. 初始化 Cutoff 函数
        cutoff_fn = CosineCutoff(cutoff_val)
        
        # 2. 实例化 GotenNet 骨干
        self.backbone = GotenNetWrapper(
            n_atom_basis=hidden_channels, 
            n_interactions=num_layers,
            cutoff_fn=cutoff_fn
        )
        
        # 3. 初始化原子级回归头 (Atom-wise Readout)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, output_dim) 
        )

    def forward(self, input_dict):
        # 兼容性解包：提取 PyG Batch
        batch = input_dict['graph'] if 'graph' in input_dict else input_dict
        
        # 防御性编程：补全原子序数
        if not hasattr(batch, 'z') or batch.z is None:
            if hasattr(batch, 'x'):
                batch.z = batch.x[:, 0].long() if batch.x.dim() > 1 else batch.x.long()
        
        # 1. 通过 GotenNet 提取原子特征
        out = self.backbone(batch)
        h = out[0] if isinstance(out, tuple) else out  # 兼容不同返回格式
        
        # 2. 原子级预测 (Atom-wise Readout) -> [N_atoms, output_dim]
        atom_preds = self.output_head(h)
        
        # 3. 分子级求和聚合 (Global Add Pool) -> [Batch_Size, output_dim]
        mol_preds = global_add_pool(atom_preds, batch.batch)
        
        return mol_preds