from typing import Callable, Dict, Union, Optional, List
import torch
from torch import nn
import schnetpack.properties as properties
from schnetpack.nn import Dense, scatter_add
from schnetpack.nn.activations import shifted_softplus

import schnetpack.nn as snn


__all__ = ["SchNet", "SchNetInteraction"]


class SchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems."""
    # 

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        activation: Callable = shifted_softplus,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            activation: if None, no activation function is used.
        """
        super(SchNetInteraction, self).__init__() # 调用父类初始化
        # 定义输入到滤波器的全连接层：将原子特征映射到滤波器维度的空间
        # 输入维度: n_atom_basis, 输出维度: n_filters, 无偏置, 无激活函数
        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        
        # 定义滤波器到输出的序列网络：将卷积后的特征映射回原子特征空间，并进行非线性变换
        self.f2out = nn.Sequential(
            # 第一层：从滤波器维度映射回原子基组维度，使用激活函数
            Dense(n_filters, n_atom_basis, activation=activation),
            # 第二层：线性变换，无激活函数，作为残差连接前的最终输出
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        
        # 定义生成滤波器的网络 (Filter Generating Network)
        # 这里的关键思想是：卷积核 W_ij 不是静态的参数，而是由原子间距离 r_ij 动态生成的
        self.filter_network = nn.Sequential(
            # 输入是径向基函数(RBF)扩展后的距离特征，输出是滤波器维度
            Dense(n_rbf, n_filters, activation=activation), 
            Dense(n_filters, n_filters)
        )

    def forward(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ):
        """Compute interaction output.

        Args:
            x: input values (原子当前的特征向量)
            f_ij: radial basis functions of distances (原子对距离的径向基函数扩展)
            idx_i: index of center atom i (中心原子索引)
            idx_j: index of neighbors j (邻居原子索引)
            rcut_ij: cutoff values (截断函数值，用于平滑过渡到0)

        Returns:
            atom features after interaction (交互后的原子特征更新值)
        """
        # 1. 线性变换：将当前原子特征 x 映射到适合卷积的维度
        x = self.in2f(x)
        
        # 2. 生成滤波器：通过 Filter Network 从径向基特征 f_ij 生成滤波器权重 Wij
        Wij = self.filter_network(f_ij)
        
        # 3. 应用截断：将滤波器权重乘以截断函数值，确保在截断半径处平滑衰减为0
        # [:, None] 用于增加维度以便广播乘法
        Wij = Wij * rcut_ij[:, None]

        # 4. 连续滤波器卷积 (Continuous-filter convolution)
        # 获取邻居原子 j 的特征
        x_j = x[idx_j]
        
        # 元素级乘法：邻居特征 * 对应的距离相关滤波器权重
        x_ij = x_j * Wij
        
        # 聚合操作：使用 scatter_add 将所有属于同一个中心原子 i 的邻居贡献 x_ij 加在一起
        # 这一步完成了卷积中的"求和"部分。dim_size 确保输出大小与原子总数一致
        x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])

        # 5. 输出变换：经过两层全连接层处理聚合后的特征
        x = self.f2out(x)
        
        # 返回的是交互带来的特征增量（稍后会在 SchNet 类中通过残差连接加到原特征上）
        return x


class SchNet(nn.Module):
    """SchNet architecture for learning representations of atomistic systems
    ... (参考文献省略) ...
    """
    # 

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        n_filters: int = None,
        shared_interactions: bool = False,
        activation: Union[Callable, nn.Module] = shifted_softplus,
        nuclear_embedding: Optional[nn.Module] = None,
        electronic_embeddings: Optional[List] = None,
    ):
        """
        Args:
           ... (参数文档省略) ...
        """
        super().__init__() # 初始化父类
        self.n_atom_basis = n_atom_basis # 原子特征向量的维度 (embedding_dim)
        # 如果未指定滤波器数量，则默认与原子基组维度相同
        self.n_filters = n_filters or self.n_atom_basis 
        self.radial_basis = radial_basis # 径向基函数层 (例如高斯展开)
        self.cutoff_fn = cutoff_fn # 截断函数层 (例如 Cosine Cutoff)
        self.cutoff = cutoff_fn.cutoff # 截断半径数值

        # 初始化嵌入层 (Embeddings)
        if nuclear_embedding is None:
            # 如果没有提供自定义嵌入，使用标准的 PyTorch Embedding
            # 输入是原子序数(最大100)，输出是维度为 n_atom_basis 的向量
            nuclear_embedding = nn.Embedding(100, n_atom_basis)
        self.embedding = nuclear_embedding
        
        # 初始化电子相关的额外嵌入 (可选，如自旋、电荷等)
        if electronic_embeddings is None:
            electronic_embeddings = []
        self.electronic_embeddings = nn.ModuleList(electronic_embeddings)



        # 初始化交互块 (Interaction Blocks)
        # 使用 replicate_module 重复创建 n_interactions 个 SchNetInteraction 模块
        # 这些模块堆叠在一起，形成深层网络
        self.interactions = snn.replicate_module(
            lambda: SchNetInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf, # RBF 的数量决定了滤波器生成网络的输入维度
                n_filters=self.n_filters,
                activation=activation,
            ),
            n_interactions,
            shared_interactions, # 是否在不同层之间共享权重
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        
        # 从输入字典中提取张量
        atomic_numbers = inputs[properties.Z] # 原子序数 (N,)
        r_ij = inputs[properties.Rij] # 原子对相对向量 (N_pairs, 3)
        idx_i = inputs[properties.idx_i] # 原子对中中心原子的索引 (N_pairs,)
        idx_j = inputs[properties.idx_j] # 原子对中邻居原子的索引 (N_pairs,)

        # 计算原子对特征
        # 计算欧几里得距离 (N_pairs,)
        d_ij = torch.norm(r_ij, dim=1) 
        # 将标量距离扩展为径向基函数特征向量 (N_pairs, n_rbf)
        f_ij = self.radial_basis(d_ij) 
        # 计算截断函数值，用于限制相互作用范围 (N_pairs,)
        rcut_ij = self.cutoff_fn(d_ij) 

        # 计算初始原子嵌入
        # 根据原子序数 Z 查找对应的 embedding 向量 x (N, n_atom_basis)
        x = self.embedding(atomic_numbers)
        # 如果有电子嵌入（如电荷、自旋），加到特征上
        for embedding in self.electronic_embeddings:
            x = x + embedding(x, inputs)

        # 循环经过所有交互块 (Interaction Blocks) 更新原子嵌入
        for interaction in self.interactions:
            # 这里的 v 是由当前层计算出的交互特征更新量
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            # 残差连接 (Residual Connection)：将更新量加回原特征
            # 类似于 ResNet，这有助于训练深层网络并保留原有信息
            x = x + v

        # 收集结果
        # 将最终学习到的标量表示（原子特征）存回输入字典
        inputs["scalar_representation"] = x

        return inputs