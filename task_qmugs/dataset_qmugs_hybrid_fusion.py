import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

class QMugsDualDataset(Dataset):

    def __init__(self, pkl_path, split='train', targets=None, grid_size=0.1, max_radius=10.0):
        super().__init__()
        self.split = split
        self.targets = targets or []
        self.grid_size = grid_size
        self.max_radius = max_radius
        
        print(f"📂 Loading QMugs Hybrid Fusion Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        self.data_list = data_dict[split]
        
        all_labels = []
        for item in self.data_list:
            lbl = [item[t] for t in self.targets]
            all_labels.append(lbl)
            
        self.labels = np.array(all_labels, dtype=np.float32)
        print(f"✅ Loaded {len(self.data_list)} samples for targets: {self.targets}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        

        ed_matrix = item['ed']
        ed_coords = torch.from_numpy(ed_matrix[:, :3]).float()
        ed_feat = torch.from_numpy(ed_matrix[:, 3].reshape(-1, 1)).float()
        

        mol_coords = torch.from_numpy(item['atom_coords']).float()
        atomic_numbers = torch.tensor(item['atom_types'], dtype=torch.long)
        

        graph_data = Data(pos=mol_coords, z=atomic_numbers)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            "graph": graph_data,
            "point_cloud": {
                "coord": ed_coords,
                "feat": ed_feat
            },
            "label": label
        }

def qmugs_dual_collate_fn(batch_list):

    graphs = []
    pc_coords, pc_feats, offsets, batch_ids = [], [], [], []
    labels = []
    
    batch_offset = 0
    
    for i, sample in enumerate(batch_list):

        graphs.append(sample["graph"])
        

        labels.append(sample["label"])
        
        pc = sample["point_cloud"]
        pc_coords.append(pc["coord"])
        pc_feats.append(pc["feat"])
        
        num_points = pc["coord"].shape[0]
        
        batch_offset += num_points
        offsets.append(batch_offset)
        
        batch_ids.append(torch.full((num_points,), i, dtype=torch.long))

    batched_graph = Batch.from_data_list(graphs)
    
    return {
        "graph": batched_graph,
        "point_cloud": {
            "coord": torch.cat(pc_coords, dim=0),
            "feat": torch.cat(pc_feats, dim=0),
            "offset": torch.tensor(offsets, dtype=torch.long),
            "batch": torch.cat(batch_ids, dim=0),
            "grid_size": 0.1
        },
        "labels": torch.stack(labels, dim=0)
    }