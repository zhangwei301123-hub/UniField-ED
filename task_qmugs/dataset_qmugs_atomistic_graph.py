import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

class QMugsAtomisticDataset(Dataset):
    """
    QMugs 纯原子图 Dataset (专为 Equiformer, SchNet, ComENet 等适配)
    """
    def __init__(self, pkl_path, split='train', targets=None, grid_size=None):
        super().__init__()
        self.split = split
        self.targets = targets or []
        
        # grid_size 在纯图模型中不使用，但为了兼容 config 参数，这里接收并忽略
        
        print(f"📂 Loading QMugs Atomistic Graph Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        # QMugs 切分后的数据是一个 List
        self.sample_list = data_dict[split]
        
        # 预先提取所有标签，方便后续转为连续的内存排布
        self.labels = []
        for item in self.sample_list:
            lbl_array = []
            for t in self.targets:
                raw_val = item[t]
                # 💡 前线轨道 (HOMO, LUMO, GAP) 不需要扣除原子参考能，直接读取即可
                lbl_array.append(raw_val)
                
            self.labels.append(lbl_array)
            
        self.labels = np.array(self.labels, dtype=np.float32)
        print(f"✅ Loaded {len(self.sample_list)} samples for targets: {self.targets}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        item = self.sample_list[idx]
        
        # ================== 提取原子图物理特征 ==================
        # 1. 坐标 (N, 3) - 确保是 float32
        mol_coords = torch.from_numpy(item['atom_coords']).float()
        
        # 2. 原子序数 (N,) - 必须是 long 类型，作为 Embedding 层的索引
        atomic_numbers = torch.tensor(item['atom_types'], dtype=torch.long)
        
        # 3. 封装为 PyG 标准 Data 对象
        graph_data = Data(pos=mol_coords, z=atomic_numbers)

        # 4. 提取对应的标签 (Target_Dim,)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            "graph": graph_data, 
            "label": label
        }

def qmugs_atomistic_collate_fn(batch_list):
    """
    纯原子图 Collate 函数：
    使用 PyG 的 Batch 将独立的分子图打包成一个包含 disconnected subgraph 的大图
    """
    # 1. 组装 Graph 数据
    graphs = [item['graph'] for item in batch_list]
    batched_graph = Batch.from_data_list(graphs)

    # 2. 组装标签数据
    labels = [item['label'] for item in batch_list]

    # 3. 返回统一字典结构，对接 engine.py 中的 input_dict
    return {
        "graph": batched_graph,
        "labels": torch.stack(labels, dim=0)
    }