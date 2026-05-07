import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch

class ED5OEAtomisticDataset(Dataset):

    def __init__(self, pkl_path, split='train', targets=None):
        super().__init__()
        self.split = split
        self.targets = targets or []
        
        self.HARTREE_TO_MEV = 27211.386245988532898
        
        self.TARGET_MAP = {
            'homo_2': 0, 'homo_1': 1, 'homo_0': 2,
            'lumo_0': 3, 'lumo_1': 4, 'lumo_2': 5, 'lumo_3': 6
        }

        print(f"📂 Loading Atomistic Graph Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        self.sample_list = data_dict[split]
        

        target_indices = [self.TARGET_MAP[t] for t in self.targets]
        
        self.labels = []
        for item in self.sample_list:
            lbl_np = np.array(item['label'], dtype=np.float32)
            lbl_np = lbl_np * self.HARTREE_TO_MEV
            self.labels.append(lbl_np[target_indices])
            
        self.labels = np.stack(self.labels).astype(np.float32)
        
        print(f"✅ Loaded {len(self.sample_list)} samples for targets: {self.targets}")
        print(f"🔄 Unit conversion: Hartree -> meV applied.")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        item = self.sample_list[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        mol_z = torch.tensor(item['mol_z'], dtype=torch.long)
        mol_pos = torch.tensor(item['mol_coords'], dtype=torch.float32)
        

        data = Data(z=mol_z, pos=mol_pos, y=label.unsqueeze(0))
        
        return data

def atomistic_collate_fn(batch):

    graph_batch = Batch.from_data_list(batch)
    
    return {
        "graph": graph_batch,
        "labels": graph_batch.y  
    }