import torch
from torch import nn
from torch.nn import Linear, Embedding 
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from math import sqrt

from ...utils import xyz_to_dat
from .features import dist_emb, angle_emb

try:
    import sympy as sym
except ImportError:
    sym = None

def swish(x):
    return x * torch.sigmoid(x)

class emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(emb, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        return dist_emb, angle_emb


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish):
        super(init, self).__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, emb, i, j):
        rbf,_ = emb
        x = self.emb(x)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size, num_spherical, num_radial, 
        num_before_skip, num_after_skip, act=swish):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf = emb
        x1,_ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2 


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init):
        super(update_v, self).__init__()
        self.act = act
        self.output_init = output_init

        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == 'zeros':
            self.lin.weight.data.fill_(0)
        if self.output_init == 'GlorotOrthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, e, i,num_nodes=None):
        _, e2 = e
        v = scatter(e2, i, dim=0, dim_size=num_nodes)
        v = self.lin_up(v)
        for lin in self.lins:
            v = self.act(lin(v))
        v = self.lin(v)
        return v


class update_u(torch.nn.Module):
    def __init__(self):
        super(update_u, self).__init__()

    def forward(self, u, v, batch):
        u += scatter(v, batch, dim=0)
        return u


class DimeNetPP(torch.nn.Module):
    r"""
    DimeNet++ 的重新实现。
    参考论文: "Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules" (arXiv:2011.14115)
    框架参考: "Spherical Message Passing for 3D Molecular Graphs" (ICLR 2021)
    
    参数说明:
        energy_and_force (bool): 是否预测能量并计算其对原子位置的负导数作为力。 [cite: 82]
        cutoff (float): 原子间交互的截断半径（单位：埃）。 [cite: 91]
        num_layers (int): 堆叠的交互块（Building Blocks）数量。 [cite: 255]
        hidden_channels (int): 隐藏层嵌入向量的维度。 [cite: 95]
        out_channels (int): 输出结果的维度（如预测单个能量标量，则为1）。 [cite: 70]
        int_emb_size (int): 交互三元组（Interaction Triplets）中使用的嵌入维度。 [cite: 257]
        basis_emb_size (int): 基函数转换时使用的嵌入维度。 [cite: 256]
        out_emb_channels (int): 输出块中原子嵌入的维度。 [cite: 262]
        num_spherical (int): 球谐函数的数量 (N_SHBF)，用于捕捉角度信息。 [cite: 171]
        num_radial (int): 径向基函数的数量 (N_RBF)，用于捕捉距离信息。 [cite: 172]
        envelop_exponent (int): 截断信封函数的指数，确保截断处平滑（两次连续可微）。 [cite: 236]
        num_before_skip/num_after_skip (int): 交互块中残差层在跳跃连接前后的数量。 [cite: 259]
        num_output_layers (int): 输出块中线性层的数量。 [cite: 262]
        act: 激活函数，DimeNet默认使用 Swish (Self-gated activation)。 [cite: 265]
        output_init: 输出层的初始化方式。
    """
    def __init__(
        self, energy_and_force=False, cutoff=5.0, num_layers=4, 
        hidden_channels=128, out_channels=1, int_emb_size=64, basis_emb_size=8, out_emb_channels=256, 
        num_spherical=7, num_radial=6, envelope_exponent=5, 
        num_before_skip=1, num_after_skip=2, num_output_layers=3, 
        act=swish, output_init='GlorotOrthogonal'):
        super(DimeNetPP, self).__init__()

        self.cutoff = cutoff
        self.energy_and_force = energy_and_force

        # 1. 初始化模块：生成初始的边嵌入 e [cite: 247, 248]
        self.init_e = init(num_radial, hidden_channels, act)
        
        # 2. 初始化输出模块：从边嵌入 e 得到原子贡献 v [cite: 261, 262]
        self.init_v = update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init)
        
        # 3. 初始化全局状态模块：聚合原子贡献得到全局属性 u (如分子总能量) [cite: 263]
        self.init_u = update_u()
        
        # 4. 物理基函数模块：生成球面贝塞尔函数和球谐函数的联合基表示 
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)
        
        # 5. 堆叠交互更新层：
        # update_vs: 更新原子层级的输出特征 [cite: 227]
        self.update_vs = torch.nn.ModuleList([
            update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init) for _ in range(num_layers)])

        # update_es: 核心交互块，利用角度和距离更新消息嵌入 [cite: 226, 255]
        self.update_es = torch.nn.ModuleList([
            update_e(
                hidden_channels, int_emb_size, basis_emb_size,
                num_spherical, num_radial,
                num_before_skip, num_after_skip,
                act,
            )
            for _ in range(num_layers)
        ])

        # update_us: 累加每一层的输出到全局结果 [cite: 228]
        self.update_us = torch.nn.ModuleList([update_u() for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        """初始化或重置模型所有层的参数"""
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()

    def forward(self, batch_data):
        # 获取原子序数、位置和批次索引
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        
        # 如果需要计算力，开启位置的梯度跟踪
        if self.energy_and_force:
            pos.requires_grad_()
            
        # 根据截断半径构建邻居图（边索引）
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        num_nodes = z.size(0)
        
        # 将笛卡尔坐标转换为物理量：距离(dist)、夹角(angle)以及索引信息 
        # idx_kj, idx_ji 用于识别构成夹角的三元组原子索引
        dist, angle, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes, use_torsion=False)

        # A. 生成基于物理的基函数嵌入 (结合了球谐函数和贝塞尔函数) 
        emb = self.emb(dist, angle, idx_kj)

        # B. 初始化阶段
        # 1. 结合原子类型 z 和距离信息生成初始边消息 e 
        e = self.init_e(z, emb, i, j)
        # 2. 将边消息聚合到节点，得到初步的原子级输出 v 
        v = self.init_v(e, i, num_nodes=num_nodes)
        # 3. 将原子输出聚合，初始化全局分子属性 u (如能量) 
        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch)

        # C. 迭代更新阶段 (消息传递)
        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            # 1. 交互块更新边消息：这是球谐函数发挥作用的地方，考虑了夹角信息 
            e = update_e(e, emb, idx_kj, idx_ji)
            # 2. 输出块更新：将更新后的消息转换为原子级输出 
            v = update_v(e, i, num_nodes=num_nodes)
            # 3. 全局累加：将本层得到的预测结果累加到总输出 u 中
            u = update_u(u, v, batch) 

        return u