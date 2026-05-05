import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

import torch_geometric
import math

from .registry import register_model
from .instance_norm import EquivariantInstanceNorm
from .graph_norm import EquivariantGraphNorm
from .layer_norm import EquivariantLayerNormV2
from .fast_layer_norm import EquivariantLayerNormFast
from .radial_func import RadialProfile
from .tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate, sort_irreps_even_first)
from .fast_activation import Activation, Gate
from .drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath
from .gaussian_rbf import GaussianRadialBasisLayer

# for bessel radial basis
#from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis
from .radial_basis import RadialBasis

_RESCALE = True
_USE_BIAS = True

# QM9
_MAX_ATOM_TYPE = 120
# Statistics of QM9 with cutoff radius = 5
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666
  

def get_norm_layer(norm_type):
    if norm_type == 'graph':
        return EquivariantGraphNorm
    elif norm_type == 'instance':
        return EquivariantInstanceNorm
    elif norm_type == 'layer':
        return EquivariantLayerNormV2
    elif norm_type == 'fast_layer':
        return EquivariantLayerNormFast
    elif norm_type is None:
        return None
    else:
        raise ValueError('Norm type {} not supported.'.format(norm_type))
    

class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope
        
    
    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2
    
    
    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)
            

def get_mul_0(irreps):
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0


class FullyConnectedTensorProductRescaleNorm(FullyConnectedTensorProductRescale):
    
    def __init__(self, irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None, norm_layer='graph'):
        
        super().__init__(irreps_in1, irreps_in2, irreps_out,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization)
        self.norm = get_norm_layer(norm_layer)(self.irreps_out)
        
        
    def forward(self, x, y, batch, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.norm(out, batch=batch)
        return out
        

class FullyConnectedTensorProductRescaleNormSwishGate(FullyConnectedTensorProductRescaleNorm):
    
    def __init__(self, irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None, norm_layer='graph'):
        
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
        super().__init__(irreps_in1, irreps_in2, gate.irreps_in,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization, norm_layer=norm_layer)
        self.gate = gate
        
        
    def forward(self, x, y, batch, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.norm(out, batch=batch)
        out = self.gate(out)
        return out
    

class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):
    
    def __init__(self, irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None):
        
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
        super().__init__(irreps_in1, irreps_in2, gate.irreps_in,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization)
        self.gate = gate
        
        
    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.gate(out)
        return out
    

def DepthwiseTensorProduct(irreps_node_input, irreps_edge_attr, irreps_node_output, 
    internal_weights=False, bias=True):
    '''
    构建一个深度（Depthwise）张量积模块。
    
    参数:
        irreps_node_input: 输入节点特征的 Irreps (例如 '128x0e+64x1e')。
        irreps_edge_attr: 边缘属性的 Irreps (通常是球谐函数)。
        irreps_node_output: 期望输出特征的 Irreps。此参数主要用作“过滤器”，
                            决定保留哪些计算路径。
        internal_weights: 是否在模块内部存储可学习的权重。
                          如果为 False，则需要在 forward 时从外部传入权重 (由 RBF 网络生成)。
        bias: 是否添加偏置项 (仅对 L=0 的标量输出有效)。
    '''
    
    irreps_output = [] # 用于存储实际生成的输出 Irreps 列表
    instructions = []  # 用于存储 TensorProduct 的指令列表 (路径定义)
    
    # --- 1. 遍历所有可能的组合 ---
    # 遍历输入的每一个不可约表示 (i: 索引, mul: 通道数, ir_in: 具体的 Irrep 如 1e)
    for i, (mul, ir_in) in enumerate(irreps_node_input):
        # 遍历边缘属性的每一个不可约表示 (通常 mul 都是 1)
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            # 计算两者张量积后可能产生的所有输出 Irrep (基于 Clebsch-Gordan 规则)
            # 例如: 1e * 1e -> 0e + 1e + 2e
            for ir_out in ir_in * ir_edge:
                
                # --- 2. 过滤路径 ---
                # 只有当生成的 ir_out 存在于用户指定的 irreps_node_output 中，
                # 或者它是标量 (0e) 时，才保留这条路径。
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    
                    # 记录当前这个新生成的输出在 irreps_output 列表中的索引
                    k = len(irreps_output)
                    
                    # 添加到输出列表
                    # 注意：这里的 mul 直接继承自输入的 mul。
                    # 这就是 "Depthwise" 的关键：输入有多少通道，输出就对应多少通道，不进行通道维度的全连接混合。
                    irreps_output.append((mul, ir_out))
                    
                    # --- 3. 构建指令 ---
                    # (i, j, k, 'uvu', True) 含义:
                    # i: 输入特征的索引
                    # j: 边缘特征的索引
                    # k: 输出特征的索引
                    # 'uvu': 连接模式。这是实现 Depthwise 的核心。
                    #        u: 输入通道数 (mul)
                    #        v: 边缘通道数 (通常为 1)
                    #        u: 输出通道数 (等于输入 mul)
                    #        这意味着每个输入通道独立计算，权重是逐通道广播的。
                    # True: 表示这条路径需要权重 (has_weight=True)
                    instructions.append((i, j, k, 'uvu', True))
    
    # 将列表转换为 e3nn 的 Irreps 对象
    irreps_output = o3.Irreps(irreps_output)
    
    # --- 4. 排序与重排 ---
    # 对输出的 Irreps 进行排序 (通常是按 L 从小到大，偶宇称在前)。
    # sort_irreps_even_first 返回排序后的 Irreps, 排列索引 p, 和逆排列索引。
    irreps_output, p, _ = sort_irreps_even_first(irreps_output) 
    
    # 更新指令中的输出索引 k。
    # 因为 irreps_output 的顺序变了，之前的索引 k 需要映射到新的位置 p[k]。
    instructions = [(i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions]
        
    # --- 5. 实例化模块 ---
    # TensorProductRescale 通常是对 e3nn.o3.TensorProduct 的封装，加入了归一化缩放。
    tp = TensorProductRescale(irreps_node_input, irreps_edge_attr,
            irreps_output, instructions,
            internal_weights=internal_weights, # 是否使用内部权重
            shared_weights=internal_weights,   # 权重是否共享
            bias=bias, rescale=_RESCALE)       # 偏置和缩放配置
            
    return tp

class SeparableFCTP(torch.nn.Module):
    '''
        Use separable FCTP for spatial convolution.
    '''
    def __init__(self, irreps_node_input, irreps_edge_attr, irreps_node_output, 
        fc_neurons, use_activation=False, norm_layer='graph', 
        internal_weights=False):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        norm = get_norm_layer(norm_layer)
        
        self.dtp = DepthwiseTensorProduct(self.irreps_node_input, self.irreps_edge_attr, 
            self.irreps_node_output, bias=False, internal_weights=internal_weights)
        
        self.dtp_rad = None
        if fc_neurons is not None:
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
            for (slice, slice_sqrt_k) in self.dtp.slices_sqrt_k.values():
                self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
                self.dtp_rad.offset.data[slice] *= slice_sqrt_k
                
        irreps_lin_output = self.irreps_node_output
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_output)
        if use_activation:
            irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
            irreps_lin_output = irreps_lin_output.simplify()
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), irreps_lin_output)
        
        self.norm = None
        if norm_layer is not None:
            self.norm = norm(self.lin.irreps_out)
        
        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0:
                gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
            else:
                gate = Gate(
                    irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                    irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                    irreps_gated  # gated tensors
                )
            self.gate = gate
    
    
    def forward(self, node_input, edge_attr, edge_scalars, batch=None, **kwargs):
        '''
            Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by 
            self.dtp_rad(`edge_scalars`).
        '''
        weight = None
        if self.dtp_rad is not None and edge_scalars is not None:    
            weight = self.dtp_rad(edge_scalars)
        out = self.dtp(node_input, edge_attr, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out
        

@compile_mode('script')
class Vec2AttnHeads(torch.nn.Module):
    '''
        Reshape vectors of shape [N, irreps_mid] to vectors of shape
        [N, num_heads, irreps_head].
    '''
    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
    
    
    def forward(self, x):
        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out
    
    
    def __repr__(self):
        return '{}(irreps_head={}, num_heads={})'.format(
            self.__class__.__name__, self.irreps_head, self.num_heads)
    
    
@compile_mode('script')
class AttnHeads2Vec(torch.nn.Module):
    '''
        Convert vectors of shape [N, num_heads, irreps_head] into
        vectors of shape [N, irreps_head * num_heads].
    '''
    def __init__(self, irreps_head):
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
    
    
    def forward(self, x):
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out
    
    
    def __repr__(self):
        return '{}(irreps_head={})'.format(self.__class__.__name__, self.irreps_head)


class ConcatIrrepsTensor(torch.nn.Module):
    
    def __init__(self, irreps_1, irreps_2):
        super().__init__()
        assert irreps_1 == irreps_1.simplify()
        self.check_sorted(irreps_1)
        assert irreps_2 == irreps_2.simplify()
        self.check_sorted(irreps_2)
        
        self.irreps_1 = irreps_1
        self.irreps_2 = irreps_2
        self.irreps_out = irreps_1 + irreps_2
        self.irreps_out, _, _ = sort_irreps_even_first(self.irreps_out) #self.irreps_out.sort()
        self.irreps_out = self.irreps_out.simplify()
        
        self.ir_mul_list = []
        lmax = max(irreps_1.lmax, irreps_2.lmax)
        irreps_max = []
        for i in range(lmax + 1):
            irreps_max.append((1, (i, -1)))
            irreps_max.append((1, (i,  1)))
        irreps_max = o3.Irreps(irreps_max)
        
        start_idx_1, start_idx_2 = 0, 0
        dim_1_list, dim_2_list = self.get_irreps_dim(irreps_1), self.get_irreps_dim(irreps_2)
        for _, ir in irreps_max:
            dim_1, dim_2 = None, None
            index_1 = self.get_ir_index(ir, irreps_1)
            index_2 = self.get_ir_index(ir, irreps_2)
            if index_1 != -1:
                dim_1 = dim_1_list[index_1]
            if index_2 != -1:
                dim_2 = dim_2_list[index_2]
            self.ir_mul_list.append((start_idx_1, dim_1, start_idx_2, dim_2))
            start_idx_1 = start_idx_1 + dim_1 if dim_1 is not None else start_idx_1
            start_idx_2 = start_idx_2 + dim_2 if dim_2 is not None else start_idx_2
          
            
    def get_irreps_dim(self, irreps):
        muls = []
        for mul, ir in irreps:
            muls.append(mul * ir.dim)
        return muls
    
    
    def check_sorted(self, irreps):
        lmax = None
        p = None
        for _, ir in irreps:
            if p is None and lmax is None:
                p = ir.p
                lmax = ir.l
                continue
            if ir.l == lmax:
                assert p < ir.p, 'Parity order error: {}'.format(irreps)
            assert lmax <= ir.l                
        
    
    def get_ir_index(self, ir, irreps):
        for index, (_, irrep) in enumerate(irreps):
            if irrep == ir:
                return index
        return -1
    
    
    def forward(self, feature_1, feature_2):
        
        output = []
        for i in range(len(self.ir_mul_list)):
            start_idx_1, mul_1, start_idx_2, mul_2 = self.ir_mul_list[i]
            if mul_1 is not None:
                output.append(feature_1.narrow(-1, start_idx_1, mul_1))
            if mul_2 is not None:
                output.append(feature_2.narrow(-1, start_idx_2, mul_2))
        output = torch.cat(output, dim=-1)
        return output
    
    
    def __repr__(self):
        return '{}(irreps_1={}, irreps_2={})'.format(self.__class__.__name__, 
            self.irreps_1, self.irreps_2)

        
@compile_mode('script')
class GraphAttention(torch.nn.Module):
    '''
        1. Message = Alpha * Value
        2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
        3. 0e -> Activation -> Inner Product -> (Alpha)
        4. (0e+1e+...) -> (Value)
    '''
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_edge_attr, irreps_node_output,
        fc_neurons,
        irreps_head, num_heads, irreps_pre_attn=None, 
        rescale_degree=False, nonlinear_message=False,
        alpha_drop=0.1, proj_drop=0.1):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        
        # Merge src and dst
        self.merge_src = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=True)
        self.merge_dst = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=False)
        
        irreps_attn_heads = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(irreps_attn_heads) #irreps_attn_heads.sort()
        irreps_attn_heads = irreps_attn_heads.simplify() 
        mul_alpha = get_mul_0(irreps_attn_heads)
        mul_alpha_head = mul_alpha // num_heads
        irreps_alpha = o3.Irreps('{}x0e'.format(mul_alpha)) # for attention score
        irreps_attn_all = (irreps_alpha + irreps_attn_heads).simplify()
        
        self.sep_act = None
        if self.nonlinear_message:
            # Use an extra separable FCTP and Swish Gate for value
            self.sep_act = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, self.irreps_pre_attn, fc_neurons, 
                use_activation=True, norm_layer=None, internal_weights=False)
            self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
            self.sep_value = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, irreps_attn_heads, fc_neurons=None, 
                use_activation=False, norm_layer=None, internal_weights=True)
            self.vec2heads_alpha = Vec2AttnHeads(o3.Irreps('{}x0e'.format(mul_alpha_head)), 
                num_heads)
            self.vec2heads_value = Vec2AttnHeads(self.irreps_head, num_heads)
        else:
            self.sep = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, irreps_attn_all, fc_neurons, 
                use_activation=False, norm_layer=None)
            self.vec2heads = Vec2AttnHeads(
                (o3.Irreps('{}x0e'.format(mul_alpha_head)) + irreps_head).simplify(), 
                num_heads)
        
        self.alpha_act = Activation(o3.Irreps('{}x0e'.format(mul_alpha_head)), 
            [SmoothLeakyReLU(0.2)])
        self.heads2vec = AttnHeads2Vec(irreps_head)
        
        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
        
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)
        
        self.proj = LinearRS(irreps_attn_heads, self.irreps_node_output)
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_input, 
                drop_prob=proj_drop)
        
        
    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, 
        batch, **kwargs):
        
        message_src = self.merge_src(node_input)
        message_dst = self.merge_dst(node_input)
        message = message_src[edge_src] + message_dst[edge_dst]
        
        if self.nonlinear_message:          
            weight = self.sep_act.dtp_rad(edge_scalars)
            message = self.sep_act.dtp(message, edge_attr, weight)
            alpha = self.sep_alpha(message)
            alpha = self.vec2heads_alpha(alpha)
            value = self.sep_act.lin(message)
            value = self.sep_act.gate(value)
            value = self.sep_value(value, edge_attr=edge_attr, edge_scalars=edge_scalars)
            value = self.vec2heads_value(value)
        else:
            message = self.sep(message, edge_attr=edge_attr, edge_scalars=edge_scalars)
            message = self.vec2heads(message)
            head_dim_size = message.shape[-1]
            alpha = message.narrow(2, 0, self.mul_alpha_head)
            value = message.narrow(2, self.mul_alpha_head, (head_dim_size - self.mul_alpha_head))
        
        # inner product
        alpha = self.alpha_act(alpha)
        alpha = torch.einsum('bik, aik -> bi', alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = alpha.unsqueeze(-1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn = value * alpha
        attn = scatter(attn, index=edge_dst, dim=0, dim_size=node_input.shape[0])
        attn = self.heads2vec(attn)
        
        if self.rescale_degree:
            degree = torch_geometric.utils.degree(edge_dst, 
                num_nodes=node_input.shape[0], dtype=node_input.dtype)
            degree = degree.view(-1, 1)
            attn = attn * degree
            
        node_output = self.proj(attn)
        
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        
        return node_output
    
    
    def extra_repr(self):
        output_str = super(GraphAttention, self).extra_repr()
        output_str = output_str + 'rescale_degree={}, '.format(self.rescale_degree)
        return output_str
                    

@compile_mode('script')
class FeedForwardNetwork(torch.nn.Module):
    '''
        Use two (FCTP + Gate)
    '''
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_node_output, irreps_mlp_mid=None,
        proj_drop=0.1):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        
        self.fctp_1 = FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_mlp_mid, 
            bias=True, rescale=_RESCALE)
        self.fctp_2 = FullyConnectedTensorProductRescale(
            self.irreps_mlp_mid, self.irreps_node_attr, self.irreps_node_output, 
            bias=True, rescale=_RESCALE)
        
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_output, 
                drop_prob=proj_drop)
            
        
    def forward(self, node_input, node_attr, **kwargs):
        node_output = self.fctp_1(node_input, node_attr)
        node_output = self.fctp_2(node_output, node_attr)
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output
    
    
@compile_mode('script')
class TransBlock(torch.nn.Module):
    '''
        1. Layer Norm 1 -> GraphAttention -> Layer Norm 2 -> FeedForwardNetwork
        2. Use pre-norm architecture
    '''
    
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_edge_attr, irreps_node_output,
        fc_neurons,
        irreps_head, num_heads, irreps_pre_attn=None, 
        rescale_degree=False, nonlinear_message=False,
        alpha_drop=0.1, proj_drop=0.1,
        drop_path_rate=0.0,
        irreps_mlp_mid=None,
        norm_layer='layer'):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input
        
        self.norm_1 = get_norm_layer(norm_layer)(self.irreps_node_input)
        self.ga = GraphAttention(irreps_node_input=self.irreps_node_input, 
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr, 
            irreps_node_output=self.irreps_node_input,
            fc_neurons=fc_neurons,
            irreps_head=self.irreps_head, 
            num_heads=self.num_heads, 
            irreps_pre_attn=self.irreps_pre_attn, 
            rescale_degree=self.rescale_degree, 
            nonlinear_message=self.nonlinear_message,
            alpha_drop=alpha_drop, 
            proj_drop=proj_drop)
        
        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0. else None
        
        self.norm_2 = get_norm_layer(norm_layer)(self.irreps_node_input)
        #self.concat_norm_output = ConcatIrrepsTensor(self.irreps_node_input, 
        #    self.irreps_node_input)
        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input, #self.concat_norm_output.irreps_out, 
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output, 
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop)
        self.ffn_shortcut = None
        if self.irreps_node_input != self.irreps_node_output:
            self.ffn_shortcut = FullyConnectedTensorProductRescale(
                self.irreps_node_input, self.irreps_node_attr, 
                self.irreps_node_output, 
                bias=True, rescale=_RESCALE)
            
            
    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, 
        batch, **kwargs):
        
        node_output = node_input
        node_features = node_input
        node_features = self.norm_1(node_features, batch=batch)
        #norm_1_output = node_features
        node_features = self.ga(node_input=node_features, 
            node_attr=node_attr, 
            edge_src=edge_src, edge_dst=edge_dst, 
            edge_attr=edge_attr, edge_scalars=edge_scalars,
            batch=batch)
        
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features
        
        node_features = node_output
        node_features = self.norm_2(node_features, batch=batch)
        #node_features = self.concat_norm_output(norm_1_output, node_features)
        node_features = self.ffn(node_features, node_attr)
        if self.ffn_shortcut is not None:
            node_output = self.ffn_shortcut(node_output, node_attr)
        
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features
        
        return node_output
    
        
class NodeEmbeddingNetwork(torch.nn.Module):
    
    def __init__(self, irreps_node_embedding, max_atom_type=_MAX_ATOM_TYPE, bias=True):
        
        super().__init__()
        self.max_atom_type = max_atom_type
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.atom_type_lin = LinearRS(o3.Irreps('{}x0e'.format(self.max_atom_type)), 
            self.irreps_node_embedding, bias=bias)
        self.atom_type_lin.tp.weight.data.mul_(self.max_atom_type ** 0.5)
        
        
    def forward(self, node_atom):
        '''
            `node_atom` is a LongTensor.
 
        '''
        # ======================================================
        node_atom_onehot = torch.nn.functional.one_hot(node_atom, self.max_atom_type).float()
        node_attr = node_atom_onehot
        node_embedding = self.atom_type_lin(node_atom_onehot)
        
        return node_embedding, node_attr, node_atom_onehot


class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num):
        super().__init__()
        self.avg_aggregate_num = avg_aggregate_num + 0.0


    def forward(self, x, index, **kwargs):
        out = scatter(x, index, **kwargs)
        out = out.div(self.avg_aggregate_num ** 0.5)
        return out
    
    
    def extra_repr(self):
        return 'avg_aggregate_num={}'.format(self.avg_aggregate_num)
    

class EdgeDegreeEmbeddingNetwork(torch.nn.Module):
    """
    边缘度嵌入网络 (Edge Degree Embedding Network)
    
    功能：
    该模块不使用输入的节点特征 (node_input) 中的具体数值，而是生成一个初始的“单位信号”，
    通过图的边进行传播和聚合。
    
    它的作用是显式地编码节点的局部几何环境密度和方向性。
    例如：
    - L=0 特征：类似于计算节点的“加权度” (Weighted Degree)，反映周围有多少邻居、距离多远。
    - L>0 特征：反映邻居分布的各向异性 (例如，邻居是否都集中在左边)。
    """
    def __init__(self, irreps_node_embedding, irreps_edge_attr, fc_neurons, avg_aggregate_num):
        """
        :param irreps_node_embedding: 节点嵌入的不可约表示 (Irreps)，例如 '128x0e+64x1e'。
        :param irreps_edge_attr: 边缘属性的 Irreps (通常是球谐函数)，用于编码方向。
        :param fc_neurons: 径向神经网络 (Radial Profile) 的隐藏层神经元列表。
        :param avg_aggregate_num: 平均聚合邻居数 (平均度)，用于归一化。
        """
        super().__init__()
        
        # --- 1. 扩展层 (Expansion) ---
        # 定义一个线性层，将标量 '1x0e' (常数 1) 映射到高维的节点嵌入空间。
        # 这一步是为了给每个节点生成一个初始的“存在感信号”或“基向量”。
        # bias=_USE_BIAS, rescale=_RESCALE 是 e3nn 特有的参数，用于数值稳定性。
        self.exp = LinearRS(o3.Irreps('1x0e'), irreps_node_embedding, 
            bias=_USE_BIAS, rescale=_RESCALE)
            
        # --- 2. 深度张量积 (Depthwise Tensor Product) ---
        # 这是核心操作：将节点特征与边缘的几何特征 (球谐函数) 进行相互作用。
        # internal_weights=False: 表示权重不是作为参数存储在这里，而是由外部 (RadialProfile) 动态生成。
        # 这实现了：Feature(Src) ⊗ SphericalHarmonics(Edge) * RadialWeights(Length)
        self.dw = DepthwiseTensorProduct(irreps_node_embedding, 
            irreps_edge_attr, irreps_node_embedding, 
            internal_weights=False, bias=False)
            
        # --- 3. 径向神经网络 (Radial Profile) ---
        # 这是一个多层感知机 (MLP)，处理边缘的标量特征 (如距离的 RBF 编码)。
        # 输出维度是 self.dw.tp.weight_numel，即张量积层所需的所有权重的总数。
        # 它的作用是根据距离动态调整特征聚合的强度。
        self.rad = RadialProfile(fc_neurons + [self.dw.tp.weight_numel])
        
        # --- 4. 权重初始化修正 ---
        # e3nn 特有的初始化技巧。
        # 遍历张量积的每个切片 (slice)，根据输入维度的平方根 (slice_sqrt_k) 缩放径向网络的最后一层权重。
        # 目的是确保张量积输出的方差在初始化时保持为 1，防止梯度爆炸或消失。
        for (slice, slice_sqrt_k) in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k
            
        # --- 5. 投影层 (Projection) ---
        # 将张量积的输出结果投影回标准的节点嵌入格式。
        # .simplify() 会清理 Irreps 格式 (例如合并相同的项)，确保格式对齐。
        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)
        
        # --- 6. 聚合层 (Scatter) ---
        # 将边缘上的特征聚合到目标节点。
        # avg_aggregate_num 用于对求和结果进行除法缩放，保持数值范围稳定 (类似 Mean pooling 但更灵活)。
        self.scale_scatter = ScaledScatter(avg_aggregate_num)
        
    
    def forward(self, node_input, edge_attr, edge_scalars, edge_src, edge_dst, batch):
        """
        前向传播
        :param node_input: 输入节点特征 (N, D)。注意：这里只用了它的形状，没用它的数值。
        :param edge_attr: 边缘的方向特征 (E, D_sh)，即球谐函数。
        :param edge_scalars: 边缘的标量特征 (E, D_rbf)，即距离编码。
        :param edge_src: 边的源节点索引 (E,)。
        :param edge_dst: 边的目标节点索引 (E,)。
        """
        
        # 1. 生成虚拟信号
        # 创建一个形状与 node_input 匹配的全 1 张量 (只取第一个通道，变成标量 1x0e)。
        # 这代表每个节点初始都发射一个强度为 1 的“信号”。
        node_features = torch.ones_like(node_input.narrow(1, 0, 1))  #[N, 1]
        
        # 2. 信号扩展
        # 将标量 1 扩展为高维特征向量。
        # 此时所有节点的特征都是相同的 (因为输入都是 1)。
        node_features = self.exp(node_features)
        
        # 3. 计算径向权重
        # 根据边的长度 (edge_scalars) 计算张量积所需的动态权重。
        # 距离不同的边会有不同的相互作用强度。
        weight = self.rad(edge_scalars)
        
        # 4. 几何相互作用 (核心步骤)
        # node_features[edge_src]: 取出源节点的特征 (所有节点都一样)。
        # edge_attr: 边的方向 (球谐函数)。
        # weight: 边的距离权重。
        # 运算结果 edge_features 包含了边的几何信息 (方向 + 距离)。
        edge_features = self.dw(node_features[edge_src], edge_attr, weight)
        
        # 5. 特征投影
        # 线性变换混合特征。
        edge_features = self.proj(edge_features)
        
        # 6. 聚合 (计算几何度)
        # 将指向同一个目标节点 (edge_dst) 的所有边的特征加起来 (并缩放)。
        # 结果 node_features 对于每个节点来说，就是它周围几何环境的“指纹”。
        # L=0 部分代表邻居密度，L>0 部分代表邻居分布的各向异性。
        node_features = self.scale_scatter(edge_features, edge_dst, dim=0, 
            dim_size=node_features.shape[0])
            
        return node_features
    


'''
class GraphAttentionTransformer(torch.nn.Module):
    def __init__(self,
        irreps_in='5x0e',
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=5.0,
        number_of_basis=128, basis_type='gaussian', fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
        drop_path_rate=0.0,
        mean=None, std=None, scale=None, atomref=None):
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.register_buffer('atomref', atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.basis_type = basis_type
        if self.basis_type == 'gaussian':
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        elif self.basis_type == 'bessel':
            self.rbf = RadialBasis(self.number_of_basis, cutoff=self.max_radius, 
                rbf={'name': 'spherical_bessel'})
        else:
            raise ValueError
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        
        self.blocks = torch.nn.ModuleList()
        self.build_blocks()
        
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE), 
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps('1x0e'), rescale=_RESCALE)) 
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)
        
        self.apply(self._init_weights)
        
        
    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
            blk = TransBlock(irreps_node_input=self.irreps_node_embedding, 
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr, 
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons, 
                irreps_head=self.irreps_head, 
                num_heads=self.num_heads, 
                irreps_pre_attn=self.irreps_pre_attn, 
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop, 
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                norm_layer=self.norm_layer)
            self.blocks.append(blk)
            
            
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
            
                          
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                or isinstance(module, GaussianRadialBasisLayer) 
                or isinstance(module, RadialBasis)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
                    
        return set(no_wd_list)
        

    def forward(self, f_in, pos, batch, node_atom, **kwargs) -> torch.Tensor:
        
        edge_src, edge_dst = radius_graph(pos, r=self.max_radius, batch=batch,
            max_num_neighbors=1000)
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec, normalize=True, normalization='component')
        
        #node_atom = node_atom.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[node_atom]
        atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom)
        edge_length = edge_vec.norm(dim=1)
        #edge_length_embedding = sin_pos_embedding(x=edge_length, 
        #    start=0.0, end=self.max_radius, number=self.number_of_basis, 
        #    cutoff=False)
        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, 
            edge_length_embedding, edge_src, edge_dst, batch)
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        
        for blk in self.blocks:
            node_features = blk(node_input=node_features, node_attr=node_attr, 
                edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh, 
                edge_scalars=edge_length_embedding, 
                batch=batch)
        
        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)
        outputs = self.head(node_features)
        outputs = self.scale_scatter(outputs, batch, dim=0)
        
        if self.scale is not None:
            outputs = self.scale * outputs

        return outputs

'''

class GraphAttentionTransformer(torch.nn.Module):
    """
    基于图注意力机制的等变Transformer模型 (Graph Attention Transformer)。
    该模型利用 E3NN 库处理 3D 旋转等变性，适用于分子建模或点云任务。
    """
    def __init__(self,
        irreps_in='5x0e',  # 输入节点的不可约表示 (Irreps)，默认为5个标量 (5x0e)
        irreps_node_embedding='128x0e+64x1e+32x2e', # 节点嵌入层的隐层特征表示 (包含标量、矢量和二阶张量)
        num_layers=6,  # Transformer 层的数量
        irreps_node_attr='1x0e', # 节点属性的表示，默认为1个标量
        irreps_sh='1x0e+1x1e+1x2e', # 球谐函数 (Spherical Harmonics) 的表示，用于编码边缘方向信息
        max_radius=5.0,  # 构建图时的最大截断半径 (Cutoff radius)
        number_of_basis=128, # 径向基函数 (RBF) 的数量
        basis_type='gaussian', # 径向基函数的类型，默认为高斯函数
        fc_neurons=[64, 64], # 全连接层 (MLP) 的中间层神经元数量列表
        irreps_feature='512x0e', # 最终输出特征的表示，这里默认为512个标量
        irreps_head='32x0e+16x1o+8x2e', # 注意力头 (Attention Head) 的特征表示
        num_heads=4, # 多头注意力的头数
        irreps_pre_attn=None, # 注意力机制前的线性变换表示 (可选)
        rescale_degree=False, # 是否对度 (degree) 进行缩放归一化
        nonlinear_message=False, # 在消息传递中是否使用非线性激活
        irreps_mlp_mid='128x0e+64x1e+32x2e', # MLP 中间层的特征表示
        norm_layer='layer', # 归一化层的类型 (如 LayerNorm)
        alpha_drop=0.2, # Alpha Dropout 的比率 (用于自注意力权重)
        proj_drop=0.0, # 投影层的 Dropout 比率
        out_drop=0.0, # 输出层的 Dropout 比率
        drop_path_rate=0.0, # Drop Path (Stochastic Depth) 的比率
        mean=None, # 目标变量的均值 (用于输出反归一化)
        std=None, # 目标变量的标准差 (用于输出反归一化)
        scale=None, # 输出的缩放因子 (通常用于原子能量求和时的缩放)
        output_dim=1,
        atomref=None): # 原子的参考能量值 (Atom Reference Energy)
        
        super().__init__() # 初始化父类 torch.nn.Module

        # --- 保存超参数 ---
        self.max_radius = max_radius # 保存最大半径
        self.number_of_basis = number_of_basis # 保存基函数数量
        self.alpha_drop = alpha_drop # 保存 alpha dropout 率
        self.proj_drop = proj_drop # 保存投影 dropout 率
        self.out_drop = out_drop # 保存输出 dropout 率
        self.drop_path_rate = drop_path_rate # 保存 drop path 率
        self.norm_layer = norm_layer # 保存归一化层类型
        self.task_mean = mean # 保存任务均值
        self.task_std = std # 保存任务标准差
        self.scale = scale # 保存缩放因子
        self.register_buffer('atomref', atomref) # 将 atomref 注册为 buffer (不作为模型参数更新，但随模型保存)

        # --- 解析不可约表示 (Irreps) ---
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) # 解析节点属性的 Irreps 对象
        self.irreps_node_input = o3.Irreps(irreps_in) # 解析节点输入的 Irreps 对象
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding) # 解析节点嵌入的 Irreps 对象
        self.lmax = self.irreps_node_embedding.lmax # 获取嵌入表示中的最大旋转阶数 l_max (例如 2e 代表 l=2)
        self.irreps_feature = o3.Irreps(irreps_feature) # 解析最终特征的 Irreps 对象
        self.num_layers = num_layers # 保存层数
        
        # 确定边缘属性 (Edge Attribute) 的 Irreps
        # 如果未指定 irreps_sh，则根据 lmax 自动生成球谐函数的 Irreps
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
            
        # 构建全连接层的神经元结构，输入层大小为基函数数量
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        
        self.irreps_head = o3.Irreps(irreps_head) # 解析 Attention Head 的 Irreps
        self.num_heads = num_heads # 保存头数
        self.irreps_pre_attn = irreps_pre_attn # 保存预注意力 Irreps
        self.rescale_degree = rescale_degree # 保存度缩放标志
        self.nonlinear_message = nonlinear_message # 保存非线性消息标志
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) # 解析 MLP 中间层 Irreps
        
        # --- 初始化子模块 ---
        
        # 节点嵌入网络：将原子类型索引转换为 irreps_node_embedding 定义的特征向量
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        
        self.basis_type = basis_type # 保存基函数类型
        # 初始化径向基函数层 (RBF)
        if self.basis_type == 'gaussian':
            # 使用高斯径向基函数
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        elif self.basis_type == 'bessel':
            # 使用贝塞尔径向基函数
            self.rbf = RadialBasis(self.number_of_basis, cutoff=self.max_radius, 
                rbf={'name': 'spherical_bessel'})
        else:
            raise ValueError # 如果类型不支持，抛出错误
            
        # 边缘度嵌入网络：结合节点嵌入、边缘属性和边缘长度，处理图的连接性信息
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        
        # 构建 Transformer 块的列表
        self.blocks = torch.nn.ModuleList() # 初始化 ModuleList
        self.build_blocks() # 调用辅助方法构建所有层
        
        # 初始化最终的归一化层
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        
        # 初始化输出的 Dropout 层
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
            
        '''
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE),
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps(f'{output_dim}x0e'), rescale=_RESCALE) 
        )    
        '''

        # 物理逻辑：只对标量 (0e) 使用 SiLU 激活，对于带方向的向量和张量传入 None (保持原样直接通过，以维持等变性)
        acts_list = [
            torch.nn.SiLU() if ir == o3.Irrep('0e') else None 
            for _, ir in o3.Irreps(self.irreps_feature)
        ]

        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE),
            Activation(self.irreps_feature, acts=acts_list),
            LinearRS(self.irreps_feature, o3.Irreps(f'{output_dim}x0e'), rescale=_RESCALE) 
        )
            
        # 聚合层：将每个节点的输出聚合（Scatter Sum/Mean）得到图级别的输出
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)
        
        # 初始化所有权重
        self.apply(self._init_weights)
        
        
    def build_blocks(self):
        """构建 Transformer 层堆叠的辅助函数"""
        for i in range(self.num_layers): # 遍历每一层
            if i != (self.num_layers - 1):
                # 如果不是最后一层，输出特征保持为节点嵌入特征大小
                irreps_block_output = self.irreps_node_embedding
            else:
                # 如果是最后一层，输出特征转换为最终特征大小 (irreps_feature)
                irreps_block_output = self.irreps_feature
            
            # 实例化一个 Transformer 块 (TransBlock)
            blk = TransBlock(irreps_node_input=self.irreps_node_embedding, # 输入特征格式
                irreps_node_attr=self.irreps_node_attr, # 节点属性格式
                irreps_edge_attr=self.irreps_edge_attr, # 边缘属性格式 (球谐函数)
                irreps_node_output=irreps_block_output, # 输出特征格式
                fc_neurons=self.fc_neurons, # 全连接层配置
                irreps_head=self.irreps_head, # 注意力头配置
                num_heads=self.num_heads, # 头数
                irreps_pre_attn=self.irreps_pre_attn, # 预注意力配置
                rescale_degree=self.rescale_degree, # 度缩放
                nonlinear_message=self.nonlinear_message, # 非线性消息
                alpha_drop=self.alpha_drop, # Dropout 配置
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid, # MLP 中间层配置
                norm_layer=self.norm_layer) # 归一化层配置
            
            self.blocks.append(blk) # 将构建好的块添加到列表中
            
            
    def _init_weights(self, m):
        """自定义权重初始化函数"""
        if isinstance(m, torch.nn.Linear): # 如果是线性层
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0) # 将偏置初始化为 0
        elif isinstance(m, torch.nn.LayerNorm): # 如果是 LayerNorm 层
            torch.nn.init.constant_(m.bias, 0) # 偏置初始化为 0
            torch.nn.init.constant_(m.weight, 1.0) # 权重 (gamma) 初始化为 1.0
            
                          
    @torch.jit.ignore # 告诉 TorchScript 忽略此函数 (通常用于训练相关的辅助功能)
    def no_weight_decay(self):
        """返回不需要应用权重衰减 (Weight Decay) 的参数名称列表"""
        no_wd_list = [] # 初始化列表
        named_parameters_list = [name for name, _ in self.named_parameters()] # 获取所有参数名
        
        # 遍历模型中的所有子模块
        for module_name, module in self.named_modules():
            # 检查模块类型是否属于通常不需要权重衰减的层 (如 Norm 层, RBF 层等)
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                or isinstance(module, GaussianRadialBasisLayer) 
                or isinstance(module, RadialBasis)):
                
                # 遍历该模块的参数
                for parameter_name, _ in module.named_parameters():
                    # 对于线性层，只跳过 bias，权重通常需要衰减 (除非另有规定)
                    # 但这里的逻辑似乎是：如果是 Linear 且是 weight，则continue（即应用衰减），
                    # 否则（如 bias 或其他层的参数）加入 no_wd_list。
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    
                    # 拼接完整的参数名
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list # 确保参数名存在
                    no_wd_list.append(global_parameter_name) # 加入列表
                    
        return set(no_wd_list) # 返回集合去重
        

    def forward(self, f_in, pos, batch, node_atom, return_node_feats=True, **kwargs) -> torch.Tensor:
        """
        前向传播函数
        :param f_in: 输入特征 (通常未使用，因为主要依赖 node_atom 生成嵌入)
        :param pos: 节点坐标 (N, 3)
        :param batch: 批次索引 (N,)，指示每个节点属于哪个图
        :param node_atom: 节点原子类型索引 (N,)
        """
        
        # --- 1. 图构建 ---
        # 根据截断半径 r 构建邻居图，返回源节点和目标节点索引
        edge_src, edge_dst = radius_graph(pos, r=self.max_radius, batch=batch,
            max_num_neighbors=1000)
            
        # 计算边向量：源节点位置 - 目标节点位置 (E, 3)
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        
        # --- 2. 特征计算 ---
        # 计算边向量的球谐函数投影，作为边的方向特征 (E, D_sh)
        # normalize=True 表示对输入向量归一化，normalization='component' 是 E3NN 的规范化方式
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec, normalize=True, normalization='component')
        
        # 节点嵌入：将原子类型转换为特征向量 (N, D)
        # atom_embedding 是主要特征，atom_attr 是属性，atom_onehot 是独热编码
        atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom)
        
        # 计算边长度 (E,)
        edge_length = edge_vec.norm(dim=1)
        
        # 对边长度进行径向基函数 (RBF) 扩展，得到距离编码 (E, n_basis)
        edge_length_embedding = self.rbf(edge_length)
        
        # 计算边缘度嵌入信息，融合了节点特征、球谐特征和距离特征
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, 
            edge_length_embedding, edge_src, edge_dst, batch)
            
        # 将原子嵌入和度嵌入相加，作为初始节点特征
        node_features = atom_embedding + edge_degree_embedding
        
        # 创建节点属性张量，这里简单地初始化为全 1 (用于后续等变操作中的标量乘法等)
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        
        # --- 3. Transformer 层堆叠 ---
        for blk in self.blocks:
            # 将特征传入每一层 Transformer Block
            # 输入包括：节点特征、节点属性、边缘索引、边缘方向特征(SH)、边缘距离特征(RBF)、批次信息
            node_features = blk(node_input=node_features, node_attr=node_attr, 
                edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh, 
                edge_scalars=edge_length_embedding, 
                batch=batch)
        
        # --- 4. 输出层 ---
        # 对最终的节点特征进行归一化
        node_features = self.norm(node_features, batch=batch)
        
        # 应用输出 Dropout (如果设置了)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)

        if return_node_feats:
            return node_features
            
        # 通过 Head (MLP) 将高维特征映射到标量输出 (如原子能量)
        outputs = self.head(node_features)
        
        # 聚合：将属于同一个图的所有节点的输出求和/平均，得到图级别的预测结果
        outputs = self.scale_scatter(outputs, batch, dim=0)
        
        # 如果设置了缩放因子，对输出进行缩放 (常用于标准化能量值)
        if self.scale is not None:
            outputs = self.scale * outputs

        return outputs # 返回最终预测结果

@register_model
def graph_attention_transformer_l2(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


@register_model
def graph_attention_transformer_nonlinear_l2(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


@register_model
def graph_attention_transformer_nonlinear_l2_e3(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+32x0o+32x1e+32x1o+16x2e+16x2o', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+8x0o+8x1e+8x1o+4x2e+4x2o', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+96x0o+96x1e+96x1o+48x2e+48x2o',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


# Equiformer, L_max = 2, Bessel radial basis, dropout = 0.2
@register_model
def graph_attention_transformer_nonlinear_bessel_l2(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='bessel',
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


# Equiformer, L_max = 2, Bessel radial basis, dropout = 0.1
@register_model
def graph_attention_transformer_nonlinear_bessel_l2_drop01(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='bessel',
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.1, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model


# Equiformer, L_max = 2, Bessel radial basis, dropout = 0.0
@register_model
def graph_attention_transformer_nonlinear_bessel_l2_drop00(irreps_in, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, **kwargs):
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='bessel',
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref)
    return model