import os
import sys
import torch
import torch.nn as nn

current_file_path = os.path.abspath(__file__)
utils_dir = os.path.dirname(current_file_path)      
task_dir = os.path.dirname(utils_dir)               
project_root = os.path.dirname(task_dir)            

if project_root not in sys.path:
    sys.path.append(project_root)
# ==============================================

class PTv3Wrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, data_dict):

        input_dict = data_dict.get('point_cloud', data_dict)
    
        out = self.backbone(input_dict)
        feat = out.feat  
        
        batch = input_dict.get('batch', None)
        
        if batch is None:

            if 'offset' in input_dict:
                offset = input_dict['offset']
                counts = torch.diff(offset, prepend=torch.tensor([0], device=offset.device))
                batch = torch.repeat_interleave(
                    torch.arange(len(counts), device=feat.device), 
                    counts
                )
            else:
                raise KeyError("❌ PTv3Wrapper 无法在输入中找到 'batch' 或 'offset'，无法执行全局池化。")
        
        from torch_scatter import scatter
        pooled_feat = scatter(feat, batch, dim=0, reduce='mean')
        
        return pooled_feat


def build_model(model_config, output_dim, normalizer=None):
    model_name = model_config.get('name')

    # ================== 1. 稀疏点云模型 PTv3 ==================
    if model_name == "PointTransformerV3":
        from models.PTv3.PointTransformerV3 import PointTransformerV3
        
        raw_backbone = PointTransformerV3(
            in_channels=model_config['in_channels'],
            enc_depths=model_config['enc_depths'],
            enc_channels=model_config['enc_channels'],
            dec_channels=model_config['dec_channels'],
            enc_patch_size=(model_config['patch_size'],)*5,
            dec_patch_size=(model_config['patch_size'],)*4,
            mlp_ratio=4, 
            qkv_bias=True, 
            enable_flash=True, 
            enable_rpe=False,
            pdnorm_ln=False, 
            cls_mode=False,
            drop_path=model_config['drop_path_rate']
        )
        
        model = PTv3Wrapper(raw_backbone)
        
        regressor = nn.Sequential(
            nn.Linear(model_config['dec_channels'][0], 32),
            nn.GELU(),
            nn.Linear(32, output_dim) 
        )
        
        return model, regressor

    # ================== 2. 双流融合模型 ==================
    elif model_name == "DualCrossattention_visnet":
        from models.Dual_crossattention.dual_crossattention_visnet import DualStreamFusionModel
        model = DualStreamFusionModel(config=model_config, output_dim=output_dim)
        regressor = nn.Identity()
        return model, regressor

    # ================== 3. 稠密模型 PointNext ==================
    elif model_name == "PointNext":
        from models.PointNext.PointNext import PointNextModel
        model = PointNextModel(config=model_config, output_dim=output_dim)
        regressor = nn.Identity()
        return model, regressor
    # ================== 4. 纯图网络模型 ViSNet ==================
    elif model_name == "ViSNet":
        from models.Visnet.visnet_model import ViSNetModel
        
        model = ViSNetModel(config=model_config, output_dim=output_dim)
        
        regressor = nn.Identity()
        
        return model, regressor
    
    # ================== 5. 基于 SchNetPack 的 SchNet ==================
    elif model_name == "SchNet":
        from models.Schnet.schnet_model import ED5SchNetModel
        
        model = ED5SchNetModel(config=model_config, output_dim=output_dim)
        
        regressor = nn.Identity()
        
        return model, regressor
    
    # ================== 6. 包含扭曲角的模型 SphereNet ==================
    elif model_name == "SphereNet":
        from models.Spherenet.spherenet_model import ED5SphereNetModel
        
        model = ED5SphereNetModel(config=model_config, output_dim=output_dim)
        regressor = nn.Identity()
        
        return model, regressor
    
    # ================== 8. 高效边图模型 ComENet ==================
    elif model_name == "ComENet":
        from models.Comenet.comenet_model import ED5ComENetModel
        
        model = ED5ComENetModel(config=model_config, output_dim=output_dim)
        regressor = nn.Identity() 
        
        return model, regressor


    
    # ================== 10. 引入夹角信息的模型 DimeNet++ ==================
    elif model_name == "DimeNetPP":
        from models.Dimenet_pp.dimenet_pp_model import ED5DimeNetPPModel
        
        model = ED5DimeNetPPModel(config=model_config, output_dim=output_dim)
        regressor = nn.Identity() 
        
        return model, regressor
    
    # ================== 11. 原子级预测模型 GotenNet ==================
    elif model_name == "GotenNet":
        from models.Gotennet.gotennet_model import ED5GotenNetModel
        
        model = ED5GotenNetModel(config=model_config, output_dim=output_dim)
        regressor = nn.Identity() 
        
        return model, regressor
    
    # ================== 12. 自定义 UniFieldNet模型 ==================
    elif model_name == "UniFieldNet":
        from models.UniFieldNet.UniFieldNet_model import ED5UniFieldNet
        
        model = ED5UniFieldNet(config=model_config, output_dim=output_dim)
        regressor = nn.Identity()
        
        return model, regressor
    
    elif model_name == "EquiformerV2":
        from models.EquiformerV2.equiformer_v2_model import ED5EquiformerV2Model
        
        model = ED5EquiformerV2Model(config=model_config, output_dim=output_dim)
        
        sphere_channels = model_config.get('sphere_channels', 64)
        regressor = nn.Sequential(
            nn.Linear(sphere_channels, sphere_channels // 2),
            nn.SiLU(),
            nn.Linear(sphere_channels // 2, output_dim) 
        )
        
    elif model_name == "Equiformer":
        from models.Equiformer.equiformer_model import ED5Equiformer
        
        model = ED5Equiformer(config=model_config, output_dim=output_dim, normalizer=normalizer)
        regressor = nn.Identity()
        
        return model, regressor
    else:
        raise ValueError(f"❌ Unsupported model: {model_name}")