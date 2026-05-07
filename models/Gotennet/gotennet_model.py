import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from gotennet import GotenNetWrapper
except ImportError as e:
    print(f"❌ 导入 GotenNetWrapper 失败，请确保 gotennet.py 在 {current_dir} 目录下。")
    raise e

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

    def __init__(self, config, output_dim=1):
        super().__init__()
        
        cutoff_val = config.get('cutoff', 10.0)
        hidden_channels = config.get('hidden_channels', 256)
        num_layers = config.get('num_layers', 8)
        

        cutoff_fn = CosineCutoff(cutoff_val)
        

        self.backbone = GotenNetWrapper(
            n_atom_basis=hidden_channels, 
            n_interactions=num_layers,
            cutoff_fn=cutoff_fn
        )
        

        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, output_dim) 
        )

    def forward(self, input_dict):

        batch = input_dict['graph'] if 'graph' in input_dict else input_dict

        if not hasattr(batch, 'z') or batch.z is None:
            if hasattr(batch, 'x'):
                batch.z = batch.x[:, 0].long() if batch.x.dim() > 1 else batch.x.long()
        

        out = self.backbone(batch)
        h = out[0] if isinstance(out, tuple) else out  # 兼容不同返回格式
        
        atom_preds = self.output_head(h)
        
        mol_preds = global_add_pool(atom_preds, batch.batch)
        
        return mol_preds