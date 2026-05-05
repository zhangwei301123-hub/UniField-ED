# utils/builder.py
import torch.nn as nn
from models.PTv3.PointTransformerV3 import PointTransformerV3


def build_model(model_config):
    """
    根据配置文件动态构建 Backbone 和 Regressor
    """
    model_name = model_config.get('name')

    # ================= 1. 构建 Backbone =================
    if model_name == "PointTransformerV3":

        backbone = PointTransformerV3(
            in_channels=model_config['in_channels'],
            enc_depths=model_config['enc_depths'],
            enc_channels=model_config['enc_channels'],
            dec_channels=model_config['dec_channels'],
            enc_patch_size=(model_config['patch_size'],)*5,
            dec_patch_size=(model_config['patch_size'],)*4,
            mlp_ratio=4, qkv_bias=True, enable_flash=True, enable_rpe=False,
            pdnorm_ln=False, cls_mode=False, drop_path=model_config['drop_path_rate']
        )
        # PTv3 的输出特征维度通常是 dec_channels 的第一个值
        head_in_channels = model_config['dec_channels'][0]

    else:
        raise ValueError(f"❌ Unsupported model architecture: {model_name}")

    # ================= 2. 构建统一的回归头 =================
    # 这样就可以根据不同骨干网络的输出维度动态调整 Linear 层的 in_features
    regressor = nn.Sequential(
        nn.Linear(head_in_channels, 32),
        nn.GELU(),
        nn.Linear(32, 1)
    )

    return backbone, regressor