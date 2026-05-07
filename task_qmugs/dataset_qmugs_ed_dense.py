import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

class QMugsDenseDataset(Dataset):

    def __init__(self, pkl_path, split='train', grid_size=0.1, targets=None):
        super().__init__()
        self.split = split
        self.grid_size = grid_size
        self.targets = targets or []
        
        print(f"📂 Loading QMugs Dense Point Cloud Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        self.data_list = data_dict[split]

        all_labels = []
        for item in self.data_list:
            # 例如 item['homo_energy_meV']
            lbl = [item[t] for t in self.targets]
            all_labels.append(lbl)
            
        self.labels = np.array(all_labels, dtype=np.float32)
        # ============================================================
            
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

def qmugs_dense_collate_fn(batch):

    batch_coord, batch_feat, batch_labels = [], [], []

    for data in batch:
        batch_coord.append(data['coord'])
        
        feat = data['feat'].transpose(0, 1)
        batch_feat.append(feat)
        
        batch_labels.append(data['label'])


    pos = torch.stack(batch_coord, dim=0).contiguous()

    x = torch.stack(batch_feat, dim=0).contiguous()

    labels = torch.stack(batch_labels, dim=0)

    return {
        "point_cloud": {"pos": pos, "x": x},
        "labels": labels
    }