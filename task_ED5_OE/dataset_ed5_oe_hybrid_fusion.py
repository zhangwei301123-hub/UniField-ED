import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch

class ED5OEDualDataset(Dataset):
    """
    ED5 双流融合 Dataset (原子图结构 + 电子密度点云)
    
    输入 pkl 要求为列表字典格式: {'train': [sample_dict, ...], ...}
    sample_dict 需包含: 'ed_points', 'mol_coords', 'mol_z', 'label'
    """
    def __init__(self, pkl_path, split='train', grid_size=0.1, max_radius=5.0, targets=None):
        super().__init__()
        self.split = split
        self.grid_size = grid_size
        self.max_radius = max_radius
        self.targets = targets or []
        
        # 定义单位转换系数: 1 Hartree = 27211.38624598853 meV
        self.HARTREE_TO_MEV = 27211.386245988532898
        
        # 建立属性映射表，对齐 (N, 7) 标签矩阵
        self.TARGET_MAP = {
            'homo_2': 0,
            'homo_1': 1,
            'homo_0': 2,
            'lumo_0': 3,
            'lumo_1': 4,
            'lumo_2': 5,
            'lumo_3': 6
        }

        print(f"📂 Loading Dual-Stream Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        self.sample_list = data_dict[split]
        
        # 解析 targets 对应的列索引
        target_indices = [self.TARGET_MAP[t] for t in self.targets]
        
        # 预先提取全局标签矩阵，方便训练循环统一计算均值和方差
        self.labels = []
        for item in self.sample_list:
            # 读取当前分子的原始 7 维标签 (Hartree)
            lbl_np = np.array(item['label'], dtype=np.float32)
            
            # [💡 核心修改] 执行单位转换 (Hartree -> meV)
            lbl_np = lbl_np * self.HARTREE_TO_MEV
            
            # 根据 YAML 配置动态切片
            lbl_filtered = lbl_np[target_indices]
            self.labels.append(lbl_filtered)
            
        self.labels = np.stack(self.labels).astype(np.float32)
        print(f"✅ Loaded {len(self.sample_list)} samples for targets: {self.targets}")
        print(f"🔄 Converted labels from Hartree to meV (Multiplier: {self.HARTREE_TO_MEV:.4f})")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        item = self.sample_list[idx]
        
        # 提取动态切片后的标签 (现在已经是 meV 单位了)
        label_val = self.labels[idx]
        label = torch.tensor(label_val, dtype=torch.float32)
        
        # ================= 1. 图数据 (Graph Modality) =================
        mol_z = torch.tensor(item['mol_z'], dtype=torch.long)
        mol_pos = torch.tensor(item['mol_coords'], dtype=torch.float32)
        # 组装为通用的 PyG Data 对象
        graph_data = Data(z=mol_z, pos=mol_pos, y=label.unsqueeze(0))
        
        # ================= 2. 点云数据 (Point Cloud Modality) =================
        raw_points = item['ed_points']
        coord = torch.from_numpy(raw_points[:, :3]).float()
        feat = torch.from_numpy(raw_points[:, 3].reshape(-1, 1)).float()
        
        pc_data = {
            "coord": coord,
            "feat": feat,
            "label": label, 
            "grid_size": self.grid_size
        }
        
        # 统一返回多模态字典
        return {
            "graph": graph_data,
            "point_cloud": pc_data,
            "label": label 
        }

def dual_collate_fn(batch):
    """
    双流通用 Collate 函数
    """
    graph_list = [sample['graph'] for sample in batch]
    pc_list = [sample['point_cloud'] for sample in batch]
    labels_list = [sample['label'] for sample in batch]
    
    # 1. 处理图网络数据 (PyG Batch)
    graph_batch = Batch.from_data_list(graph_list)
    
    # 2. 处理点云数据 (Flattened Offset)
    coords, feats, offsets = [], [], []
    batch_offset = 0
    pc_batch_idx = [] 
    
    for i, sample in enumerate(pc_list):
        coords.append(sample["coord"])
        feats.append(sample["feat"])
        
        n_points = sample["coord"].shape[0]
        batch_offset += n_points
        offsets.append(batch_offset)
        
        pc_batch_idx.append(torch.full((n_points,), i, dtype=torch.long))
        
    pc_batch = {
        "coord": torch.cat(coords, dim=0),
        "feat": torch.cat(feats, dim=0),
        "offset": torch.tensor(offsets, dtype=torch.long),
        "batch": torch.cat(pc_batch_idx, dim=0), 
        "grid_size": pc_list[0]["grid_size"]
    }
    
    # 3. 统一全局标签
    labels_batch = torch.stack(labels_list, dim=0)
    
    return {
        "graph": graph_batch,
        "point_cloud": pc_batch,
        "labels": labels_batch
    }