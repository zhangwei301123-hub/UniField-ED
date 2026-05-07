import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch

class ED5OEDualDataset(Dataset):

    def __init__(self, pkl_path, split='train', grid_size=0.1, max_radius=5.0, targets=None):
        super().__init__()
        self.split = split
        self.grid_size = grid_size
        self.max_radius = max_radius
        self.targets = targets or []
        
        
        self.HARTREE_TO_MEV = 27211.386245988532898
        
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
        
        target_indices = [self.TARGET_MAP[t] for t in self.targets]

        self.labels = []
        for item in self.sample_list:

            lbl_np = np.array(item['label'], dtype=np.float32)
            
            lbl_np = lbl_np * self.HARTREE_TO_MEV
            
            lbl_filtered = lbl_np[target_indices]
            self.labels.append(lbl_filtered)
            
        self.labels = np.stack(self.labels).astype(np.float32)
        print(f"✅ Loaded {len(self.sample_list)} samples for targets: {self.targets}")
        print(f"🔄 Converted labels from Hartree to meV (Multiplier: {self.HARTREE_TO_MEV:.4f})")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        item = self.sample_list[idx]
        
        label_val = self.labels[idx]
        label = torch.tensor(label_val, dtype=torch.float32)
    
        mol_z = torch.tensor(item['mol_z'], dtype=torch.long)
        mol_pos = torch.tensor(item['mol_coords'], dtype=torch.float32)

        graph_data = Data(z=mol_z, pos=mol_pos, y=label.unsqueeze(0))
        
        raw_points = item['ed_points']
        coord = torch.from_numpy(raw_points[:, :3]).float()
        feat = torch.from_numpy(raw_points[:, 3].reshape(-1, 1)).float()
        
        pc_data = {
            "coord": coord,
            "feat": feat,
            "label": label, 
            "grid_size": self.grid_size
        }
        
        return {
            "graph": graph_data,
            "point_cloud": pc_data,
            "label": label 
        }

def dual_collate_fn(batch):

    graph_list = [sample['graph'] for sample in batch]
    pc_list = [sample['point_cloud'] for sample in batch]
    labels_list = [sample['label'] for sample in batch]
    
    graph_batch = Batch.from_data_list(graph_list)
    
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
    
    labels_batch = torch.stack(labels_list, dim=0)
    
    return {
        "graph": graph_batch,
        "point_cloud": pc_batch,
        "labels": labels_batch
    }