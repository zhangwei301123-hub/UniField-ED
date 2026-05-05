import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

class ED5OEEDPointCloudDataset(Dataset):
    """
    专门用于加载 EDBench 预处理后的 .pkl 点云数据
    """
    def __init__(self, pkl_path, split='train', grid_size=0.1):
        super().__init__()
        self.split = split
        self.grid_size = grid_size
        
        # 定义单位转换常数 (Hartree -> meV)
        # 1 Ha = 27.2114 eV = 27211.386245988532898 meV
        self.HARTREE_TO_MEV = 1

        print(f"📂 Loading data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        # 根据预处理代码逻辑，提取 coords 和 labels
        self.coords_list = data_dict[split]['coords'] 
        
        # [关键修改] 获取标签并立即进行单位转换
        raw_labels = data_dict[split]['labels']
        
        # 确保是 Numpy 数组以便进行向量化乘法
        if isinstance(raw_labels, list):
            raw_labels = np.array(raw_labels)
            
        # 执行转换：Hartree -> meV
        self.labels = raw_labels * self.HARTREE_TO_MEV
        
        print(f"✅ Loaded {len(self.coords_list)} samples.")
        print(f"🔄 Labels converted from Hartree to meV (Factor: {self.HARTREE_TO_MEV})")

    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        # 获取原始数据 (N, 4) -> x, y, z, density
        raw_data = self.coords_list[idx]
        
        # 分离坐标和特征
        # coord: (N, 3)
        coord = raw_data[:, :3] 
        # feat: (N, 1) -> 只取最后一列密度值，并增加一个维度
        feat = raw_data[:, 3].reshape(-1, 1) 
        
        # 获取标签 (此时已经是 meV 单位了)
        label = self.labels[idx]

        # 转换为 Tensor
        coord = torch.from_numpy(coord).float()
        feat = torch.from_numpy(feat).float()
        label = torch.tensor(label).float()

        return {
            "coord": coord, 
            "feat": feat, 
            "label": label
        }

def ptv3_collate_fn(batch):
    """
    [关键] 自定义 Collate 函数
    将一个 Batch 的数据"拍扁" (Flatten)，并生成 offset
    """
    coords = []
    feats = []
    labels = []
    offsets = []
    
    batch_offset = 0
    
    for sample in batch:
        coords.append(sample["coord"])
        feats.append(sample["feat"])
        labels.append(sample["label"])
        
        # 更新 offset: 当前所有点的累计数量
        batch_offset += sample["coord"].shape[0]
        offsets.append(batch_offset)
        
    # 拼接 (Flatten)
    # shape: (Total_Points, 3)
    coord_batch = torch.cat(coords, dim=0)
    # shape: (Total_Points, 1)
    feat_batch = torch.cat(feats, dim=0)
    # shape: (Batch_Size, Label_Dim)
    label_batch = torch.stack(labels, dim=0)
    # shape: (Batch_Size,)
    offset_batch = torch.tensor(offsets).long()
    
    return {
        "coord": coord_batch,
        "feat": feat_batch,
        "offset": offset_batch,
        "labels": label_batch 
    }