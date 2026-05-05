import torch
import torch.nn as nn
from torch_cluster import radius  
from torch_scatter import scatter
from torch_geometric.utils import softmax as pyg_softmax
import os
import sys

# ================== 1. 引入依赖 ==================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 坚决去掉 try-except，让所有的导入错误“大声报错”
from visnet.models.visnet_block import ViSNetBlock 
from visnet.models.output_modules import EquivariantScalar 

# PTv3 根据我们在 builder.py 里的习惯，通常直接从根目录的 models.PTv3 导入
from models.PTv3.PointTransformerV3 import PointTransformerV3

# ================== 2. 核心组件定义 ==================

class CloudToAtomInteraction(nn.Module):
    """
    [VisField-Net 核心组件]
    实现从 PTv3 电子云特征到 VisNet 原子特征的 Cross-Attention。
    """
    def __init__(self, atom_dim, cloud_dim, interaction_radius=2.0, num_heads=4):
        super().__init__()
        self.atom_dim = atom_dim
        self.cloud_dim = cloud_dim
        self.interaction_radius = interaction_radius
        
        self.q_atom = nn.Linear(atom_dim, atom_dim)
        self.k_cloud = nn.Linear(cloud_dim, atom_dim)
        self.v_cloud = nn.Linear(cloud_dim, atom_dim)
        
        self.out_proj = nn.Linear(atom_dim, atom_dim)
        self.norm = nn.LayerNorm(atom_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_atom, pos_atom, batch_atom, x_cloud, pos_cloud, batch_cloud):
        # 1. 空间索引 (Bipartite Graph)
        edge_index = radius(x=pos_cloud, y=pos_atom, 
                            r=self.interaction_radius,
                            batch_x=batch_cloud, batch_y=batch_atom,
                            max_num_neighbors=64) 
        
        idx_atom, idx_cloud = edge_index[0], edge_index[1]
        
        # 2. 准备 Q, K, V
        Q = self.q_atom(x_atom)      
        K = self.k_cloud(x_cloud)    
        V = self.v_cloud(x_cloud)    
        
        # 3. 稀疏 Attention 计算
        q_vec = Q[idx_atom]          
        k_vec = K[idx_cloud]         
        v_vec = V[idx_cloud]         
        
        scores = (q_vec * k_vec).sum(dim=-1) / (self.atom_dim ** 0.5)
        attn_weights = pyg_softmax(scores, idx_atom)
        
        # 4. 聚合 (Aggregation)
        msg = v_vec * attn_weights.unsqueeze(-1)
        aggr_feat = scatter(msg, idx_atom, dim=0, dim_size=x_atom.size(0), reduce='sum')
        
        # 5. 残差连接与更新
        x_atom_new = self.norm(x_atom + self.dropout(self.out_proj(self.act(aggr_feat))))
        
        return x_atom_new


class DualStreamFusionModel(nn.Module):
    """
    [修改点]: 移除了 __init__ 中的 mean 和 std 参数。
    统一接收 config 字典和动态的 output_dim。
    """
    def __init__(self, config, output_dim=1):
        super().__init__()
        
        # 从模型配置中剥离出子配置
        visnet_args = config.get('visnet', {})
        ptv3_args = config.get('ptv3', {})
        
        # --- A. 初始化 ViSNet ---
        hidden_channels = visnet_args.get('hidden_channels', 128)
        
        self.visnet_dim = hidden_channels
        self.visnet_rep = ViSNetBlock(
            lmax=visnet_args.get('lmax', 2),
            vecnorm_type=visnet_args.get('vecnorm_type', 'none'),
            trainable_vecnorm=visnet_args.get('trainable_vecnorm', False),
            num_heads=visnet_args.get('num_heads', 8),
            num_layers=visnet_args.get('num_layers', 6),
            hidden_channels=hidden_channels, 
            num_rbf=visnet_args.get('num_rbf', 32),
            rbf_type=visnet_args.get('rbf_type', 'expnorm'),
            trainable_rbf=visnet_args.get('trainable_rbf', False),
            activation=visnet_args.get('activation', 'silu'),
            attn_activation=visnet_args.get('attn_activation', 'silu'),
            max_z=visnet_args.get('max_z', 100),
            cutoff=visnet_args.get('cutoff', 5.0),
            max_num_neighbors=visnet_args.get('max_num_neighbors', 32),
            vertex_type=visnet_args.get('vertex_type', 'Edge')
        )
        
        # [修改点]: 将动态推导出的 output_dim (例如 7) 传给 ViSNet 的等变输出模块
        self.visnet_out = EquivariantScalar(
            hidden_channels=self.visnet_dim,
            output_dim=output_dim
        )
        
        # --- B. 初始化 PTv3 ---
        enc_depths = ptv3_args.get('enc_depths', [2, 2, 2, 6, 2])
        num_stages = len(enc_depths)
        patch_size_base = ptv3_args.get('patch_size', 1024)
        
        stride = ptv3_args.get('stride', (2,) * (num_stages - 1))
        dec_depths = ptv3_args.get('dec_depths', (2,) * (num_stages - 1))
        enc_patch_size = (patch_size_base,) * num_stages
        dec_patch_size = (patch_size_base,) * (num_stages - 1)
        
        enc_num_head = ptv3_args.get('enc_num_head', [2**(i+1) for i in range(num_stages)])
        dec_num_head = ptv3_args.get('dec_num_head', [4 * (2**i) for i in range(num_stages - 1)])

        self.ptv3 = PointTransformerV3(
            in_channels=ptv3_args.get('in_channels', 1),
            enc_depths=enc_depths,
            enc_channels=ptv3_args.get('enc_channels', [32, 64, 128, 256, 512]),
            dec_channels=ptv3_args.get('dec_channels', [64, 64, 128, 256]),
            stride=stride,
            dec_depths=dec_depths,
            enc_patch_size=enc_patch_size,
            dec_patch_size=dec_patch_size,
            enc_num_head=enc_num_head,
            dec_num_head=dec_num_head,
            mlp_ratio=4, qkv_bias=True, enable_flash=True, enable_rpe=False,
            pdnorm_ln=False, cls_mode=False 
        )
        
        self.ptv3_dim = ptv3_args.get('dec_channels', [64, 64, 128, 256])[0]
        
        # --- C. 初始化交互层 ---
        self.interaction = CloudToAtomInteraction(
            atom_dim=self.visnet_dim,
            cloud_dim=self.ptv3_dim,
            interaction_radius=config.get('interaction_radius', 2.0) 
        )
        
    def forward(self, input_dict):
        """
        [修改点]: 统一接收 input_dict，并从我们在 dataset 中命名的专业字段中解包数据。
        """
        visnet_batch = input_dict['graph']
        ptv3_dict = input_dict['point_cloud']
        
        # 1. VisNet 特征提取
        x_atom, vec_atom = self.visnet_rep(visnet_batch)
        
        # 2. PTv3 特征提取
        ptv3_out = self.ptv3(ptv3_dict)
        x_cloud = ptv3_out.feat
        
        # 3. 跨域交互
        pos_atom = visnet_batch.pos
        batch_atom = visnet_batch.batch
        pos_cloud = ptv3_out.coord
        batch_cloud = self.offset2batch(ptv3_out.offset)
        
        x_atom_refined = self.interaction(
            x_atom=x_atom,
            pos_atom=pos_atom,
            batch_atom=batch_atom,
            x_cloud=x_cloud,
            pos_cloud=pos_cloud,
            batch_cloud=batch_cloud
        )
        
        # 4. 预测与读出 (Readout)
        pred_atom = self.visnet_out.pre_reduce(
            x_atom_refined, vec_atom, visnet_batch.z, visnet_batch.pos, visnet_batch.batch
        )
        
        out = scatter(pred_atom, visnet_batch.batch, dim=0, reduce='add')
        out = self.visnet_out.post_reduce(out)
        
        # [修改点]: 移除了 out = out * std + mean。
        # 让网络直接输出归一化尺度的 logits，与 train_one_epoch 中的 criterion(pred, norm_labels) 对齐！
        return out

    def offset2batch(self, offset):
        return torch.cat([
            torch.tensor([i] * (o - (offset[i-1] if i > 0 else 0)), device=offset.device)
            for i, o in enumerate(offset)
        ])