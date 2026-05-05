import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

class ED5OEEDPointCloudDataset(Dataset):
    def __init__(self, pkl_path, split='train', grid_size=0.1, targets=None):
        super().__init__()
        self.split = split
        self.grid_size = grid_size
        self.targets = targets or []
        
        # 【关键】建立属性名到矩阵列索引的映射表
        # 注意：你需要根据生成 pkl 时打包这 7 个属性的具体顺序，来修改这里的索引数字
        self.TARGET_MAP = {
            'homo_2': 0,
            'homo_1': 1,
            'homo_0': 2,  
            'lumo_0': 3,   
            'lumo_1': 4,
            'lumo_2': 5,
            'lumo_3': 6
        }

        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        self.coords_list = data_dict[split]['coords'] 
        raw_labels = data_dict[split]['labels']  # 提取出来的形状是 (N, 7)
        
        # 根据 YAML 传进来的 targets 列表，提取对应的列
        # 例如，如果 targets=['homo_meV', 'lumo_meV']，就会提取 raw_labels[:, [2, 3]]
        target_indices = [self.TARGET_MAP[t] for t in self.targets]
        self.labels = raw_labels[:, target_indices]

    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        raw_data = self.coords_list[idx]
        
        coord = raw_data[:, :3] 
        feat = raw_data[:, 3].reshape(-1, 1) 
        label = self.labels[idx]

        return {
            "coord": torch.from_numpy(coord).float(), 
            "feat": torch.from_numpy(feat).float(), 
            "label": torch.from_numpy(label).float()
        }

# ptv3_collate_fn 保持你原有的逻辑完全不变
def ptv3_collate_fn(batch):
    coords, feats, labels, offsets = [], [], [], []
    batch_offset = 0
    
    for sample in batch:
        coords.append(sample["coord"])
        feats.append(sample["feat"])
        labels.append(sample["label"])
        
        batch_offset += sample["coord"].shape[0]
        offsets.append(batch_offset)
        
    return {
        "coord": torch.cat(coords, dim=0),
        "feat": torch.cat(feats, dim=0),
        "offset": torch.tensor(offsets, dtype=torch.long),
        "labels": torch.stack(labels, dim=0) 
    }