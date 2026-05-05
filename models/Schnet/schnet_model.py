import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_add
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# 适配 schnetpack 的属性定义

import src.schnetpack.nn as snn
import src.schnetpack.properties as properties
from src.schnetpack.representation import SchNet

class ED5SchNetModel(nn.Module):
    """
    [SchNetPack 适配版]
    将 Lightning 代码中的逻辑迁移到通用解耦架构中。
    """
    def __init__(self, config, output_dim=1):
        super().__init__()
        self.cutoff = config.get('cutoff', 10.0)
        self.max_neighbors = config.get('max_neighbors', 32)
        hidden_channels = config.get('hidden_channels', 128)

        # 1. 初始化 SchNetPack 核心组件
        radial_basis = snn.GaussianRBF(
            n_rbf=config.get('n_rbf', 300), 
            cutoff=self.cutoff
        )
        cutoff_fn = snn.CosineCutoff(cutoff=self.cutoff)
        
        self.model = SchNet(
            n_atom_basis=hidden_channels,
            n_interactions=config.get('n_interactions', 5),
            radial_basis=radial_basis,
            cutoff_fn=cutoff_fn,
            n_filters=hidden_channels,
        )
        
        # 2. 输出层 (Readout)
        self.out_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            snn.ShiftedSoftplus(),
            nn.Linear(hidden_channels // 2, output_dim)
        )

    def forward(self, input_dict):
        # 1. 剥离出 PyG 的 Batch 对象
        batch = input_dict['graph'] if 'graph' in input_dict else input_dict
        
        # 2. 【关键转换】将 PyG Batch 转换为 SchNet 所需的 inputs 字典
        # 动态计算邻居列表和相对距离
        edge_index = radius_graph(
            batch.pos, 
            r=self.cutoff, 
            batch=batch.batch, 
            max_num_neighbors=self.max_neighbors
        )
        
        idx_j, idx_i = edge_index[0], edge_index[1]
        r_ij = batch.pos[idx_j] - batch.pos[idx_i]

        schnet_inputs = {
            properties.Z: batch.z,
            properties.Rij: r_ij,
            properties.idx_i: idx_i,
            properties.idx_j: idx_j,
        }

        # 3. 提取原子特征
        results = self.model(schnet_inputs)
        atom_feats = results["scalar_representation"]
        
        # 4. 聚合原子特征得到分子表示 (Sum Pooling)
        mol_feats = scatter_add(atom_feats, batch.batch, dim=0)
        
        # 5. 回归预测
        out = self.out_net(mol_feats)
        return out