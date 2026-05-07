import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

class QM9DenseDataset(Dataset):

    def __init__(self, pkl_path, split='train', grid_size=0.1, targets=None):
        super().__init__()
        self.split = split
        self.grid_size = grid_size
        self.targets = targets or []

        print(f"📂 Loading QM9 Dense Point Cloud Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        self.sample_list = list(data_dict[split].values())
        
        # 预先提取全局标签矩阵
        self.labels = []
        for item in self.sample_list:
            # 动态获取 targets
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
            "label": label
        }

def pointnext_collate_fn(batch):

    batch_coord, batch_feat, batch_labels = [], [], []

    for data in batch:
        batch_coord.append(data['coord'])
        # [N, C] -> [C, N] (PointNext 核心需求)
        feat = data['feat'].transpose(0, 1)
        batch_feat.append(feat)
        batch_labels.append(data['label'])

    pos = torch.stack(batch_coord, dim=0).contiguous()
    x = torch.stack(batch_feat, dim=0).contiguous()
    labels = torch.stack(batch_labels, dim=0)

    # 完美适配通用 engine.py
    return {
        "point_cloud": {"pos": pos, "x": x},
        "labels": labels
    }