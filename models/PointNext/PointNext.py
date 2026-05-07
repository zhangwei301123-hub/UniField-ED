import os
import sys
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))

if current_dir not in sys.path:
    sys.path.append(current_dir)



from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig


class PointNextModel(nn.Module):

    def __init__(self, config, output_dim):
        super().__init__()
        

        backbone_cfg = EasyConfig()
        backbone_cfg.update({
            'NAME': 'PointNextEncoder',

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
            

            'aggr_args': {'feature_type': 'dp_fj', 'reduction': 'max'},
            'group_args': {'NAME': 'ballquery', 'normalize_dp': True},
            'conv_args': {'order': 'conv-norm-act'},
            'act_args': {'act': 'relu'},
            'norm_args': {'norm': 'bn'},
        })

        self.backbone = build_model_from_cfg(backbone_cfg)


        self.width = backbone_cfg['width']
        self.out_channels = self.width * 16
        

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
            nn.Linear(hidden_dim // 2, output_dim) 
        )

    def forward(self, input_dict):

        pc_data = input_dict.get('point_cloud', input_dict)
        
        
        pos = pc_data.get('pos', pc_data.get('coord'))
        x = pc_data.get('x', pc_data.get('feat'))
        

        openpoints_input = {'pos': pos, 'x': x}

        if hasattr(self.backbone, 'forward_cls_feat'):
            feat = self.backbone.forward_cls_feat(openpoints_input)
        else:
            feat = self.backbone(openpoints_input)
            
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]

        if feat.dim() == 3:
            feat = torch.max(feat, dim=-1)[0]
            
        return self.head(feat)