import torch
import torch.nn as nn
import os
import sys
import numpy as np
import math


if not hasattr(np, 'math'):
    np.math = math


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dig.threedgraph.method import ComENet

class ED5ComENetModel(nn.Module):
    """
    [ComENet 适配版]
    基于 DIG 库实现，适配多任务回归架构。
    """
    def __init__(self, config, output_dim=1):
        super().__init__()
        
        # 实例化 ComENet
        # 它内部集成了 Global Pooling 和 Readout，直接输出到 output_dim
        self.model = ComENet(
            cutoff=config.get('cutoff', 10.0),             
            num_layers=config.get('num_layers', 5),       
            hidden_channels=config.get('hidden_channels', 128), 
            out_channels=output_dim,      # 动态匹配任务数 (如 7)
            num_radial=config.get('num_radial', 6),       
            num_spherical=config.get('num_spherical', 3),
            num_output_layers=3                           
        )

    def forward(self, input_dict):
        # 兼容性解包：从解耦字典中提取 PyG Batch 对象
        batch = input_dict['graph'] if 'graph' in input_dict else input_dict
        
        # 自动补齐原子序数 z (如果 dataset 传的是 x)
        if not hasattr(batch, 'z') and hasattr(batch, 'x'):
            batch.z = batch.x[:, 0].long() if batch.x.dim() > 1 else batch.x.long()

        # ComENet 接收完整的 PyG Batch
        out = self.model(batch)
        
        return out