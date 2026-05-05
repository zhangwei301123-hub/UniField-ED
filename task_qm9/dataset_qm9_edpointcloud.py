import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

class QM9EDPointCloudDataset(Dataset):
    """
    QM9 稀疏点云 Dataset (专为 PTv3 等稀疏模型适配)
    """
    def __init__(self, pkl_path, split='train', grid_size=0.1, targets=None):
        super().__init__()
        self.split = split
        self.grid_size = grid_size
        self.targets = targets or []

        print(f"📂 Loading QM9 Sparse Point Cloud Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        self.sample_list = list(data_dict[split].values())
        
        self.labels = []
        for item in self.sample_list:
            lbl_array = [item[t] for t in self.targets]
            self.labels.append(lbl_array)
            
        self.labels = np.array(self.labels, dtype=np.float32)
        print(f"✅ Loaded {len(self.sample_list)} samples for targets: {self.targets}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        item = self.sample_list[idx]
        
        raw_ed = item['ed']
        coord = torch.from_numpy(raw_ed[:, :3]).float()
        feat = torch.from_numpy(raw_ed[:, 3].reshape(-1, 1)).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            "coord": coord, 
            "feat": feat, 
            "label": label,
            "grid_size": self.grid_size
        }

def ptv3_collate_fn(batch):
    """
    稀疏点云 Collate 函数：
    将点云沿第 0 维度展平合并 (cat)，并生成 PTv3 必须的 offset 和 batch 索引
    """
    coords, feats, offsets = [], [], []
    batch_offset = 0
    batch_idx = [] 
    labels = []

    for i, data in enumerate(batch):
        coords.append(data['coord'])
        feats.append(data['feat'])
        labels.append(data['label'])
        
        n_points = data['coord'].shape[0]
        batch_offset += n_points
        offsets.append(batch_offset)
        
        # [💡 核心] 生成 PTv3 报缺失的那个 batch 键
        batch_idx.append(torch.full((n_points,), i, dtype=torch.long))

    ptv3_data = {
        "coord": torch.cat(coords, dim=0),
        "feat": torch.cat(feats, dim=0),
        "offset": torch.tensor(offsets, dtype=torch.long),
        "batch": torch.cat(batch_idx, dim=0), 
        "grid_size": batch[0]["grid_size"]
    }

    return {
        "point_cloud": ptv3_data,
        "labels": torch.stack(labels, dim=0)
    }