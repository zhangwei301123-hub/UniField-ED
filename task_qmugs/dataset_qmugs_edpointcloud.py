import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

class QMugsPointCloudDataset(Dataset):
    """
    QMugs 点云 Dataset (适配 PointTransformerV3 等使用 offset 的模型)
    """
    def __init__(self, pkl_path, split='train', grid_size=0.1, targets=None):
        super().__init__()
        self.split = split
        self.grid_size = grid_size
        self.targets = targets or []
        
        print(f"📂 Loading QMugs Point Cloud Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        # 根据 split 提取对应的列表数据
        self.data_list = data_dict[split]
        
        # ================= 动态提取标签 =================
        # 根据传入的 targets (如 ['homo_energy_meV', 'dipole_moment']) 动态构建
        all_labels = []
        for item in self.data_list:
            lbl = [item[t] for t in self.targets]
            all_labels.append(lbl)
            
        # 预先打包成 numpy 数组
        self.labels = np.array(all_labels, dtype=np.float32)
        # ================================================
        
        print(f"✅ Loaded {len(self.data_list)} samples for targets: {self.targets}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 提取点云及特征，你的 QMugs 数据存放在 'ed' 键里
        ed_matrix = item['ed']
        
        coord = torch.from_numpy(ed_matrix[:, :3]).float()
        feat = torch.from_numpy(ed_matrix[:, 3].reshape(-1, 1)).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            "coord": coord, 
            "feat": feat, 
            "label": label
        }

def ptv3_collate_fn(batch_list):
    coords, feats, labels, offsets, batch_ids = [], [], [], [], []
    batch_offset = 0
    
    for i, sample in enumerate(batch_list):
        coords.append(sample["coord"])
        feats.append(sample["feat"])
        labels.append(sample["label"])
        
        num_points = sample["coord"].shape[0]
        
        # 记录 offset
        batch_offset += num_points
        offsets.append(batch_offset)
        
        # 【💡 核心修复】生成每个点对应的 batch 索引 (例如当前是第 i 个样本，就生成 num_points 个 i)
        batch_ids.append(torch.full((num_points,), i, dtype=torch.long))
        
    return {
        "coord": torch.cat(coords, dim=0),
        "feat": torch.cat(feats, dim=0),
        "offset": torch.tensor(offsets, dtype=torch.long),
        "batch": torch.cat(batch_ids, dim=0),  # <--- 解决 AssertionError 的关键
        "grid_size": 0.1,                      # PTv3 通常需要知道 grid_size 用于后续 spconv 
        "labels": torch.stack(labels, dim=0) 
    }