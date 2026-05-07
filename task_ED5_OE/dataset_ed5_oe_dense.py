import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

class ED5OEDenseDataset(Dataset):

    def __init__(self, pkl_path, split='train', grid_size=0.1, targets=None, convert_to_mev=False):
        super().__init__()
        self.split = split
        self.grid_size = grid_size
        self.targets = targets or []
        
        self.convert_to_mev = convert_to_mev
        self.HARTREE_TO_MEV = 27211.386245988532898
        
        self.TARGET_MAP = {
            'homo_2': 0, 'homo_1': 1, 'homo_0': 2,
            'lumo_0': 3, 'lumo_1': 4, 'lumo_2': 5, 'lumo_3': 6
        }

        print(f"📂 Loading Dense Point Cloud Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        self.coords_list = data_dict[split]['coords'] 
        raw_labels = data_dict[split]['labels'] 
        
        target_indices = [self.TARGET_MAP[t] for t in self.targets]
        self.labels = raw_labels[:, target_indices]
        
        if self.convert_to_mev:
            self.labels = self.labels * self.HARTREE_TO_MEV
            print(f"🔄 Converted labels from Hartree to meV")
            
        print(f"✅ Loaded {len(self.coords_list)} samples for targets: {self.targets}")

    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        raw_data = self.coords_list[idx]
        
        coord = torch.from_numpy(raw_data[:, :3]).float()
        feat = torch.from_numpy(raw_data[:, 3].reshape(-1, 1)).float()
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

        feat = data['feat'].transpose(0, 1)
        batch_feat.append(feat)
        
        batch_labels.append(data['label'])


    pos = torch.stack(batch_coord, dim=0).contiguous()
    # [B, C, N]
    x = torch.stack(batch_feat, dim=0).contiguous()
    # [B, num_targets]
    labels = torch.stack(batch_labels, dim=0)

    return {
        "point_cloud": {"pos": pos, "x": x},
        "labels": labels
    }