import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

class QMugsPointCloudDataset(Dataset):

    def __init__(self, pkl_path, split='train', grid_size=0.1, targets=None):
        super().__init__()
        self.split = split
        self.grid_size = grid_size
        self.targets = targets or []
        
        print(f"📂 Loading QMugs Point Cloud Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        self.data_list = data_dict[split]
        

        all_labels = []
        for item in self.data_list:
            lbl = [item[t] for t in self.targets]
            all_labels.append(lbl)
            
        self.labels = np.array(all_labels, dtype=np.float32)
        # ================================================
        
        print(f"✅ Loaded {len(self.data_list)} samples for targets: {self.targets}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

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
        
       
        batch_ids.append(torch.full((num_points,), i, dtype=torch.long))
        
    return {
        "coord": torch.cat(coords, dim=0),
        "feat": torch.cat(feats, dim=0),
        "offset": torch.tensor(offsets, dtype=torch.long),
        "batch": torch.cat(batch_ids, dim=0),  
        "grid_size": 0.1,                      
        "labels": torch.stack(labels, dim=0) 
    }