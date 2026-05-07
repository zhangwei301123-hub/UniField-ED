import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

class QMugsAtomisticDataset(Dataset):

    def __init__(self, pkl_path, split='train', targets=None, grid_size=None):
        super().__init__()
        self.split = split
        self.targets = targets or []
        
        
        print(f"📂 Loading QMugs Atomistic Graph Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        self.sample_list = data_dict[split]
        
        self.labels = []
        for item in self.sample_list:
            lbl_array = []
            for t in self.targets:
                raw_val = item[t]
                lbl_array.append(raw_val)
                
            self.labels.append(lbl_array)
            
        self.labels = np.array(self.labels, dtype=np.float32)
        print(f"✅ Loaded {len(self.sample_list)} samples for targets: {self.targets}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        item = self.sample_list[idx]
        
        mol_coords = torch.from_numpy(item['atom_coords']).float()
        
        atomic_numbers = torch.tensor(item['atom_types'], dtype=torch.long)
        
        graph_data = Data(pos=mol_coords, z=atomic_numbers)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            "graph": graph_data, 
            "label": label
        }

def qmugs_atomistic_collate_fn(batch_list):

    graphs = [item['graph'] for item in batch_list]
    batched_graph = Batch.from_data_list(graphs)

    labels = [item['label'] for item in batch_list]

    return {
        "graph": batched_graph,
        "labels": torch.stack(labels, dim=0)
    }