import torch
import torch.nn as nn
import os
import sys
import numpy as np
import math

# ================== 💡 核心补丁：修复 NumPy 兼容性 ==================
if not hasattr(np, 'math'):
    np.math = math
# ===================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dig.threedgraph.method import DimeNetPP

class ED5DimeNetPPModel(nn.Module):
    """
    [DimeNet++ 适配版]
    基于 DIG 库实现，适配多任务回归架构。
    """
    def __init__(self, config, output_dim=1):
        super().__init__()
        
        # 实例化 DimeNet++
        # 它内部自带 Global Pooling 和 Readout MLP
        self.model = DimeNetPP(
            energy_and_force=False, # 属性预测，不计算力
            cutoff=config.get('cutoff', 5.0),
            num_layers=config.get('num_layers', 4),
            hidden_channels=config.get('hidden_channels', 128),
            out_channels=output_dim,  # 动态匹配任务数 (如 7)
            int_emb_size=config.get('int_emb_size', 64),
            basis_emb_size=config.get('basis_emb_size', 8),
            out_emb_channels=config.get('out_emb_channels', 256),
            num_spherical=config.get('num_spherical', 7),
            num_radial=config.get('num_radial', 6),
            envelope_exponent=config.get('envelope_exponent', 5),
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3
        )

    def forward(self, input_dict):
        # 兼容性解包：从解耦字典中提取 PyG Batch 对象
        batch = input_dict['graph'] if 'graph' in input_dict else input_dict
        
        # 自动补齐原子序数 z (防止 dataset 没有按标准传参)
        if not hasattr(batch, 'z') or batch.z is None:
            if hasattr(batch, 'x'):
                batch.z = batch.x[:, 0].long() if batch.x.dim() > 1 else batch.x.long()
            else:
                raise ValueError("DimeNet++ 需要原子序数特征 (z 或 x)")

        # 直接传入 PyG Batch
        out = self.model(batch)
        
        return out