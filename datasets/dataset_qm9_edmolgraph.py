import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch


class QM9EDMolGraphDataset(Dataset):
    """
    针对 QM9 数据集的双流融合 Dataset (ViSNet + PTv3)
    """
    def __init__(self, pkl_path, split='train', grid_size=0.1, target_keys=None):
        super().__init__()
        self.split = split
        self.grid_size = grid_size
        
        if target_keys is None:
            self.target_keys = ['homo_meV', 'lumo_meV', 'gap_meV']
        else:
            self.target_keys = target_keys

        print(f"📂 Loading QM9 Dual-Stream Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        # ================== 【核心修复点】 ==================
        raw_split_data = data_dict[split]
        
        # 检查它是不是字典 (比如 {"mol_1": {...}, "mol_2": {...}})
        # 如果是，我们只需要它里面的 values 组成一个列表
        if isinstance(raw_split_data, dict):
            self.sample_list = list(raw_split_data.values())
        else:
            self.sample_list = raw_split_data
        # ====================================================
        
        # 提取选定的多任务标签
        self.labels = []
        for item in self.sample_list:
  
            label_vector = [item[k] for k in self.target_keys]
            self.labels.append(label_vector)
            
        self.labels = np.array(self.labels, dtype=np.float32)
        print(f"✅ Loaded {len(self.sample_list)} samples. Target keys: {self.target_keys}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        item = self.sample_list[idx]
        
        # ================= ViSNet 数据 (Structure) =================
        # 键名更新: 'atom_types' 和 'atom_coords'
        mol_z = torch.tensor(item['atom_types'], dtype=torch.long)
        mol_pos = torch.tensor(item['atom_coords'], dtype=torch.float32)
        
        # 提取标签向量
        label_val = [item[k] for k in self.target_keys]
        label = torch.tensor(label_val, dtype=torch.float32)
        
        visnet_data = Data(z=mol_z, pos=mol_pos, y=label.unsqueeze(0))
        
        # ================= PTv3 数据 (Density Point Cloud) =================
        # 键名更新: 'ed'
        raw_points = item['ed']
        
        coord = torch.from_numpy(raw_points[:, :3]).float()
        feat = torch.from_numpy(raw_points[:, 3].reshape(-1, 1)).float()
        
        ptv3_data = {
            "coord": coord,
            "feat": feat,
            "label": label, 
            "grid_size": self.grid_size
        }
        
        return {
            "visnet": visnet_data,
            "ptv3": ptv3_data,
            "label": label 
        }
        

def dual_stream_collate_fn(batch):
    """
    双流 Collate 函数：
    1. 使用 PyG 的 Batch.from_data_list 处理 ViSNet 数据
    2. 使用 PTv3 的 Flatten 逻辑处理点云数据
    """
    
    # 1. 分离数据
    visnet_list = [sample['visnet'] for sample in batch]
    ptv3_list = [sample['ptv3'] for sample in batch]
    labels_list = [sample['label'] for sample in batch]
    
    # 2. 处理 ViSNet (Structure)
    # PyG 会自动处理 batch 索引 (data.batch)
    visnet_batch = Batch.from_data_list(visnet_list)
    
    # 3. 处理 PTv3 (Density)
    coords = []
    feats = []
    offsets = []
    batch_offset = 0
    
    # 构建 batch index (可选，用于 scatter pooling)
    ptv3_batch_idx = [] 
    
    for i, sample in enumerate(ptv3_list):
        coords.append(sample["coord"])
        feats.append(sample["feat"])
        
        n_points = sample["coord"].shape[0]
        batch_offset += n_points
        offsets.append(batch_offset)
        
        ptv3_batch_idx.append(torch.full((n_points,), i, dtype=torch.long))
        
    ptv3_batch = {
        "coord": torch.cat(coords, dim=0),
        "feat": torch.cat(feats, dim=0),
        "offset": torch.tensor(offsets).long(),
        "batch": torch.cat(ptv3_batch_idx, dim=0), # PTv3 融合时需要这个做 scatter
        "grid_size": ptv3_list[0]["grid_size"]
    }
    
    # 4. 统一标签
    labels_batch = torch.stack(labels_list, dim=0)
    
    return {
        "visnet": visnet_batch,
        "ptv3": ptv3_batch,
        "labels": labels_batch
    }