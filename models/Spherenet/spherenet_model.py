import torch
import torch.nn as nn
import os
import sys
import numpy as np
import math
if not hasattr(np, 'math'):
    np.math = math


# 💡 动态挂载你指定的 DIG 路径
# 既然 dig 文件夹就在当前目录(Spherenet)下，我们直接挂载当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 确保能导入 dig.threedgraph.method
from dig.threedgraph.method import SphereNet

class ED5SphereNetModel(nn.Module):
    """
    [SphereNet 适配版]
    基于 DIG 库实现，适配多任务回归架构。
    """
    def __init__(self, config, output_dim=1):
        super().__init__()
        
        # 实例化 SphereNet
        # 它内部集成了 Global Pooling 和 Readout，直接输出到 output_dim
        self.model = SphereNet(
            energy_and_force=False, # 属性预测任务
            cutoff=config.get('cutoff', 5.0),
            num_layers=config.get('num_layers', 4),
            hidden_channels=config.get('hidden_channels', 128),
            out_channels=output_dim,  # 动态匹配任务数 (如 7)
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
        # 兼容性解包：从解耦字典中提取 PyG Batch 对象
        batch = input_dict['graph'] if 'graph' in input_dict else input_dict
        
        # SphereNet 接收完整的 PyG Batch，内部处理 z, pos 和 batch 索引
        out = self.model(batch)
        
        return out