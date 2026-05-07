import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

class QM9DualDataset(Dataset):

    def __init__(self, pkl_path, split='train', grid_size=0.1, targets=None, max_radius=5.0):
        super().__init__()
        self.split = split
        self.grid_size = grid_size
        self.targets = targets or []
        self.max_radius = max_radius

        print(f"📂 Loading QM9 Dual-Stream Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        split_data = data_dict[split]
        if isinstance(split_data, dict):
            self.sample_list = list(split_data.values())
        elif isinstance(split_data, list):
            self.sample_list = split_data
        else:
            raise TypeError(f"❌ 数据格式错误: data_dict['{split}'] 既不是 dict 也不是 list，而是 {type(split_data)}")
        # =========================================================================
        
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
        pc_coord = torch.from_numpy(raw_ed[:, :3]).float()
        pc_feat = torch.from_numpy(raw_ed[:, 3].reshape(-1, 1)).float()
        
        mol_coords = torch.from_numpy(item['atom_coords']).float()
        atomic_numbers = torch.tensor(item['atom_types'], dtype=torch.long)
        
        graph_data = Data(pos=mol_coords, z=atomic_numbers)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            "graph": graph_data, 
            "pc_coord": pc_coord, 
            "pc_feat": pc_feat, 
            "label": label,
            "grid_size": self.grid_size
        }

def qm9_dual_collate_fn(batch_list):

    graphs = [item['graph'] for item in batch_list]
    batched_graph = Batch.from_data_list(graphs)

    coords, feats, offsets = [], [], []
    batch_offset = 0
    labels = []

    for item in batch_list:
        coords.append(item['pc_coord'])
        feats.append(item['pc_feat'])
        labels.append(item['label'])
        
        n_points = item['pc_coord'].shape[0]
        batch_offset += n_points
        offsets.append(batch_offset)

    ptv3_data = {
        "coord": torch.cat(coords, dim=0),
        "feat": torch.cat(feats, dim=0),
        "offset": torch.tensor(offsets, dtype=torch.long),
        "grid_size": batch_list[0]["grid_size"]
    }

    return {
        "graph": batched_graph,
        "point_cloud": ptv3_data,
        "labels": torch.stack(labels, dim=0)
    }