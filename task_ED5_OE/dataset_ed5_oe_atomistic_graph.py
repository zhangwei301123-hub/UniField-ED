import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch

class ED5OEAtomisticDataset(Dataset):
    """
    ED5 原子图结构 Dataset (专为 ViSNet, EquiformerV2 等图网络适配)
    
    输入 pkl 要求包含: 'mol_coords', 'mol_z', 'label'
    """
    def __init__(self, pkl_path, split='train', targets=None):
        super().__init__()
        self.split = split
        self.targets = targets or []
        
        # 1 Hartree = 27211.3862 meV
        self.HARTREE_TO_MEV = 27211.386245988532898
        
        # 属性映射表
        self.TARGET_MAP = {
            'homo_2': 0, 'homo_1': 1, 'homo_0': 2,
            'lumo_0': 3, 'lumo_1': 4, 'lumo_2': 5, 'lumo_3': 6
        }

        print(f"📂 Loading Atomistic Graph Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        self.sample_list = data_dict[split]
        
        # 解析 targets 对应的列索引
        target_indices = [self.TARGET_MAP[t] for t in self.targets]
        
        # 预先处理并提取标签矩阵
        self.labels = []
        for item in self.sample_list:
            lbl_np = np.array(item['label'], dtype=np.float32)
            # 单位转换
            lbl_np = lbl_np * self.HARTREE_TO_MEV
            # 动态切片
            self.labels.append(lbl_np[target_indices])
            
        self.labels = np.stack(self.labels).astype(np.float32)
        
        print(f"✅ Loaded {len(self.sample_list)} samples for targets: {self.targets}")
        print(f"🔄 Unit conversion: Hartree -> meV applied.")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        item = self.sample_list[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # 提取原子序数 (z) 和 3D 坐标 (pos)
        mol_z = torch.tensor(item['mol_z'], dtype=torch.long)
        mol_pos = torch.tensor(item['mol_coords'], dtype=torch.float32)
        
        # 组装为标准的 PyG Data 对象
        # y 存储标签，便于某些 PyG 原生模型直接调用
        data = Data(z=mol_z, pos=mol_pos, y=label.unsqueeze(0))
        
        return data

def atomistic_collate_fn(batch):
    """
    原子图专用 Collate 函数：将多个 Data 对象打包成一个 Batch 对象
    """
    # 使用 PyG 提供的 Batch 类进行高效合并
    graph_batch = Batch.from_data_list(batch)
    
    # 为了适配你那个“模态无关”的 engine.py，我们将 labels 单独提取出来
    # 这样 engine.py 里的 batch_data.get("labels") 就能直接拿到张量
    return {
        "graph": graph_batch,
        "labels": graph_batch.y  # 形状为 [Batch_Size, Target_Dim]
    }