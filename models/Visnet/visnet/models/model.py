import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_warn
from torch import Tensor
from torch.autograd import grad
from torch_geometric.data import Data
from torch_scatter import scatter

from visnet import priors
from visnet.models import output_modules
from visnet.models.utils import ExpNormalSmearing, GaussianSmearing, VecLayerNorm

'''
def create_model(args, prior_model=None, mean=None, std=None):
    """
    模型工厂函数：根据参数字典 args 创建 ViSNet 模型实例。
    
    Args:
        args: 包含所有模型超参数的字典 (通常来自 yaml 或 argparse)。
        prior_model: (可选) 先验模型实例，用于给预测加一个物理基准 (如 Atomref)。
        mean: (可选) 训练集标签的均值，用于输出反归一化。
        std: (可选) 训练集标签的标准差，用于输出反归一化。
    
    Returns:
        初始化好的 ViSNet 模型 (nn.Module)。
    """
    # 1. 提取 ViSNetBlock 所需的参数
    visnet_args = dict(
        lmax=args["lmax"],              # 球谐函数最大阶数 (1=Vector, 2=Tensor)
        vecnorm_type=args["vecnorm_type"], # 向量归一化类型 (max_min 或 none)
        trainable_vecnorm=args["trainable_vecnorm"], # 是否可训练向量归一化参数
        num_heads=args["num_heads"],    # 多头注意力的头数
        num_layers=args["num_layers"],  # GNN 层数 (Interaction Blocks 数量)
        hidden_channels=args["embedding_dimension"], # 隐藏层特征维度
        num_rbf=args["num_rbf"],        # 径向基函数 (RBF) 的数量
        rbf_type=args["rbf_type"],      # RBF 类型 (如 expnorm, gaussian)
        trainable_rbf=args["trainable_rbf"], # RBF 参数是否可训练
        activation=args["activation"],  # 激活函数 (如 silu)
        attn_activation=args["attn_activation"], # 注意力机制中的激活函数
        max_z=args["max_z"],            # 最大原子序数 (用于 Embedding 表大小)
        cutoff=args["cutoff"],          # 截断半径 (单位 Å)
        max_num_neighbors=args["max_num_neighbors"], # 每个原子最大邻居数
        vertex_type=args["vertex_type"], # 顶点特征类型 (Edge/Node/None)
    )

    # 2. 创建表示网络 (Representation Network)
    # 目前主要支持 ViSNetBlock，这是提取几何特征的核心骨干
    if args["model"] == "ViSNetBlock":
        from visnet.models.visnet_block import ViSNetBlock
        representation_model = ViSNetBlock(**visnet_args)
    else:
        raise ValueError(f"Unknown model {args['model']}.")
    
    # 3. 创建先验模型 (Prior Model)
    # 如果配置中指定了先验模型 (如 Atomref)，但没有传入实例，则在这里实例化
    if args["prior_model"] and prior_model is None:
        # 确保 args 里有先验模型的参数
        assert "prior_args" in args, (
            f"Requested prior model {args['prior_model']} but the "
            f'arguments are lacking the key "prior_args".'
        )
        # 检查 priors 模块里是否有这个类
        assert hasattr(priors, args["prior_model"]), (
            f'Unknown prior model {args["prior_model"]}. '
            f'Available models are {", ".join(priors.__all__)}'
        )
        # 实例化先验模型 (例如 Atomref，用于计算分子的基准能量)
        prior_model = getattr(priors, args["prior_model"])(**args["prior_args"])

    # 4. 创建输出网络 (Output Network)
    # 负责将提取的特征映射到最终的物理量 (Scalar, Vector 等)
    # 默认前缀 "Equivariant"，例如 "EquivariantScalar"
    output_prefix = "Equivariant"
    # 动态加载 Output Module 类
    output_class = getattr(output_modules, output_prefix + args["output_model"])
    
    # 实例化 Output Module
    # 注意：这里可能需要传入 output_dim 参数 (如果 args["output_model"] 支持的话)
    # 对于多任务回归 (如 7 个轨道能量)，通常需要 output_dim=7
    # 如果 args 中包含 output_dim 且 OutputModel 支持，最好在这里传入
    # 目前代码只传了 embedding_dimension 和 activation
    output_model = output_class(args["embedding_dimension"], args["activation"])

    # 5. 组装最终的 ViSNet 模型
    model = ViSNet(
        representation_model,
        output_model,
        prior_model=prior_model,
        reduce_op=args["reduce_op"], # 聚合操作 (add/mean)
        mean=mean,                   # 数据均值
        std=std,                     # 数据标准差
        derivative=args["derivative"], # 是否计算导数 (力)
    )
    return model

'''
def create_model(args, prior_model=None, mean=None, std=None):
    """
    模型工厂函数：根据参数字典 args 创建 ViSNet 模型实例。
    """
    # 1. 提取 ViSNetBlock 所需的参数
    visnet_args = dict(
        lmax=args["lmax"],              # 球谐函数最大阶数
        vecnorm_type=args["vecnorm_type"], 
        trainable_vecnorm=args["trainable_vecnorm"], 
        num_heads=args["num_heads"],    # 多头注意力的头数
        num_layers=args["num_layers"],  # GNN 层数
        hidden_channels=args["embedding_dimension"], # 隐藏层特征维度
        num_rbf=args["num_rbf"],        # 径向基函数 (RBF) 的数量
        rbf_type=args["rbf_type"],      # RBF 类型
        trainable_rbf=args["trainable_rbf"], 
        activation=args["activation"],  # 激活函数
        attn_activation=args["attn_activation"], 
        max_z=args["max_z"],            # 最大原子序数
        cutoff=args["cutoff"],          # 截断半径
        max_num_neighbors=args["max_num_neighbors"], 
        vertex_type=args["vertex_type"], 
    )

    # 2. 创建表示网络 (Representation Network)
    if args["model"] == "ViSNetBlock":
        from visnet.models.visnet_block import ViSNetBlock
        representation_model = ViSNetBlock(**visnet_args)
    else:
        raise ValueError(f"Unknown model {args['model']}.")
    
    # 3. 创建先验模型 (Prior Model)
    if args["prior_model"] and prior_model is None:
        assert "prior_args" in args, (
            f"Requested prior model {args['prior_model']} but the "
            f'arguments are lacking the key "prior_args".'
        )
        assert hasattr(priors, args["prior_model"]), (
            f'Unknown prior model {args["prior_model"]}. '
            f'Available models are {", ".join(priors.__all__)}'
        )
        prior_model = getattr(priors, args["prior_model"])(**args["prior_args"])

    # 4. 创建输出网络 (Output Network) - [修改重点区域]
    output_prefix = "Equivariant"
    output_class = getattr(output_modules, output_prefix + args["output_model"])
    
    # [新增逻辑] 自动推断输出维度
    # 默认维度为 1
    computed_output_dim = 1
    # 如果传入了 mean 且维度大于 1 (例如 [1, 7])，则使用该维度
    if mean is not None:
        if mean.ndim > 1:
            computed_output_dim = mean.shape[-1]
        elif mean.ndim == 1 and mean.shape[0] > 1:
            # 这种情况较少见 (因为 mean 通常是 [1, N])，但也防范一下
            computed_output_dim = mean.shape[0]
            
    # [修改逻辑] 实例化 Output Module
    # 我们假设 output_modules 中的类 (如 EquivariantScalar) 构造函数支持 'output_dim' 参数
    # 如果你的底层库不支持该参数，这里会报错，需要去修改 visnet/models/output_modules.py
    output_model = output_class(
        args["embedding_dimension"], 
        args["activation"],
        output_dim=computed_output_dim # <--- 关键修改：传入计算出的维度
    )

    # 5. 组装最终的 ViSNet 模型
    model = ViSNet(
        representation_model,
        output_model,
        prior_model=prior_model,
        reduce_op=args["reduce_op"],
        mean=mean,
        std=std,
        derivative=args["derivative"],
    )
    return model

def load_model(filepath, args=None, device="cpu", **kwargs):
    """
    从 checkpoint 文件加载模型
    """
    # 加载 checkpoint
    ckpt = torch.load(filepath, map_location="cpu")
    
    # 如果没有传入 args，则使用 checkpoint 里保存的超参数
    if args is None:
        args = ckpt["hyper_parameters"]

    # 允许通过 kwargs 覆盖 checkpoint 里的参数
    for key, value in kwargs.items():
        if not key in args:
            rank_zero_warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    # 创建模型结构
    model = create_model(args)
    
    # 加载权重 (去除 "model." 前缀，适应 Lightning 的命名习惯)
    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    
    return model.to(device)


class ViSNet(nn.Module):
    """
    ViSNet 主模型类
    它是 Representation + Output + Prior 的容器，并处理前向传播和梯度计算。
    """
    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
    ):
        super(ViSNet, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model

        # 处理先验模型
        self.prior_model = prior_model
        # 如果输出模型不支持先验 (例如预测偶极矩向量)，则强制禁用
        if not output_model.allow_prior_model and prior_model is not None:
            self.prior_model = None
            rank_zero_warn(
                "Prior model was given but the output model does "
                "not allow prior models. Dropping the prior model."
            )

        self.reduce_op = reduce_op
        self.derivative = derivative

        # 注册均值和标准差为 buffer (不参与梯度更新，但随模型保存/移动)
        # 如果没有提供，默认 mean=0, std=1 (即不做处理)
        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        self.reset_parameters()

    def reset_parameters(self):
        """重置所有子模块的参数"""
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(self, data: Data) -> Tuple[Tensor, Optional[Tensor]]:
        """
        前向传播逻辑
        
        Args:
            data: PyG Batch 数据对象
            
        Returns:
            (output, derivative) 元组
            - output: 预测值 (如能量)
            - derivative: 导数 (如力)，如果不计算则为 None
        """
        
        # 1. 开启坐标的梯度追踪 (为了计算力 Forces = -dE/dR)
        if self.derivative:
            data.pos.requires_grad_(True)

        # 2. 提取特征 (Representation)
        # x: 标量特征 [N_atoms, Hidden_dim]
        # v: 向量特征 [N_atoms, Hidden_dim, 3]
        x, v = self.representation_model(data)
        
        # 3. 输出头处理 (Output Pre-reduction)
        # 将高维特征映射到输出维度，通常这时候还在原子级别
        x = self.output_model.pre_reduce(x, v, data.z, data.pos, data.batch)
        
        # 4. 反标准化 (Denormalize Std) - 在求和之前乘回标准差
        # 这样做的目的是让 Prior Model (通常是物理真实的基准值) 能和网络输出在同一量级相加
        x = x * self.std

        # 5. 加上先验值 (Prior)
        # 例如：网络预测的是 "原子能量相对于基准值的偏差"，这里把基准值加回去
        if self.prior_model is not None:
            x = self.prior_model(x, data.z)

        # 6. 聚合 (Scatter Reduce)
        # 将原子级别的预测值聚合为分子级别 (通常是求和 add)
        # out: [Batch_size, Output_dim]
        out = scatter(x, data.batch, dim=0, reduce=self.reduce_op)
        
        # 7. 输出头后处理 (Post-reduction)
        # 有些任务可能需要在聚合后做处理
        out = self.output_model.post_reduce(out)
        
        # 8. 反标准化 (Denormalize Mean) - 加回均值
        out = out + self.mean

        # 9. 计算导数 (力)
        if self.derivative:
            # 计算 out 对 data.pos 的梯度
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],              # 目标函数 (能量)
                [data.pos],         # 自变量 (坐标)
                grad_outputs=grad_outputs,
                create_graph=True,  # 创建计算图以支持高阶导数
                retain_graph=True,  # 保留图
            )[0]
            
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            
            # 力是能量梯度的负数: F = -∇E
            return out, -dy
            
        # 如果不需要导数，返回 None
        return out, None