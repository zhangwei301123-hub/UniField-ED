import os
import sys
import torch
import torch.nn as nn

# ================== 1. 动态挂载路径 ==================
# 获取当前 PointNext.py 所在的绝对路径 (/home/zw/ED_all/models/PointNext)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 强行将其加入系统环境变量，这样 Python 就能认出 openpoints 文件夹了
if current_dir not in sys.path:
    sys.path.append(current_dir)

# ================== 2. 引入 OpenPoints 依赖 ==================

from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig


class PointNextModel(nn.Module):
    """
    PointNext 回归模型封装 (集成到多模态解耦框架)
    """
    def __init__(self, config, output_dim):
        super().__init__()
        
        # ================= 1. 构建 Backbone 配置 =================
        backbone_cfg = EasyConfig()
        backbone_cfg.update({
            'NAME': 'PointNextEncoder',
            
            # 从 YAML 配置中读取参数，若无则使用 PointNext-B 默认值
            'blocks': config.get('blocks', [1, 2, 4, 2, 1]),
            'width': config.get('width', 64),
            'drop_path_rate': config.get('drop_path_rate', 0.1),
            'strides': config.get('strides', [1, 2, 2, 2, 2]),
            'sa_layers': config.get('sa_layers', 2),
            'sa_use_res': config.get('sa_use_res', True),
            'in_channels': config.get('in_channels', 1),
            'expansion': config.get('expansion', 4),
            'radius': config.get('radius', 0.1),
            'nsample': config.get('nsample', 32),
            
            # 固定操作参数
            'aggr_args': {'feature_type': 'dp_fj', 'reduction': 'max'},
            'group_args': {'NAME': 'ballquery', 'normalize_dp': True},
            'conv_args': {'order': 'conv-norm-act'},
            'act_args': {'act': 'relu'},
            'norm_args': {'norm': 'bn'},
        })
        
        # 实例化 Backbone
        self.backbone = build_model_from_cfg(backbone_cfg)

        # ================= 2. 动态推导输出维度 =================
        # PointNext 的规律：经过 4 次下采样，最终通道数 = width * 16
        self.width = backbone_cfg['width']
        self.out_channels = self.width * 16
        
        # ================= 3. 构建多任务回归头 =================
        hidden_dim = self.out_channels // 2 
        
        self.head = nn.Sequential(
            nn.Linear(self.out_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.5)), 
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.5)),
            nn.Linear(hidden_dim // 2, output_dim) # 动态接收任务数量
        )

    def forward(self, input_dict):
        # 适配框架输入：如果是在双流下，点云数据在 input_dict['point_cloud'] 中
        # 如果是单模态，直接就是 input_dict
        pc_data = input_dict.get('point_cloud', input_dict)
        
        # 适配键名：你的新 collate_fn 叫 'pos' 和 'x'，兼容老叫法 'coord' 和 'feat'
        pos = pc_data.get('pos', pc_data.get('coord'))
        x = pc_data.get('x', pc_data.get('feat'))
        
        # 组装 OpenPoints 需要的字典
        openpoints_input = {'pos': pos, 'x': x}
        
        # Backbone 提取特征
        if hasattr(self.backbone, 'forward_cls_feat'):
            feat = self.backbone.forward_cls_feat(openpoints_input)
        else:
            feat = self.backbone(openpoints_input)
            
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
            
        # 全局池化
        if feat.dim() == 3:
            feat = torch.max(feat, dim=-1)[0]
            
        # 回归头预测
        return self.head(feat)