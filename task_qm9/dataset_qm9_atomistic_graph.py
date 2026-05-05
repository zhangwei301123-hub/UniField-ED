import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

class QM9AtomisticDataset(Dataset):
    """
    QM9 纯原子图 Dataset (专为 ViSNet, SchNet, GotenNet 等只需分子几何结构的模型适配)
    """
    def __init__(self, pkl_path, split='train', targets=None):
        super().__init__()
        self.split = split
        self.targets = targets or []

        print(f"📂 Loading QM9 Atomistic Graph Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        # ================== 💡 自适应 List/Dict 格式 ==================
        split_data = data_dict[split]
        if isinstance(split_data, dict):
            self.sample_list = list(split_data.values())
        elif isinstance(split_data, list):
            self.sample_list = split_data
        else:
            raise TypeError(f"❌ 数据格式错误: data_dict['{split}'] 既不是 dict 也不是 list，而是 {type(split_data)}")
        # ===============================================================
        
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
        
        # ================== 仅提取原子图数据 ==================
        # 使用你 pkl 文件中真实的键名 (Bohr 单位坐标)
        mol_coords = torch.from_numpy(item['atom_coords']).float()
        atomic_numbers = torch.tensor(item['atom_types'], dtype=torch.long)
        
        # 构建单张图的 PyG Data 对象
        graph_data = Data(pos=mol_coords, z=atomic_numbers)

        # 提取标签数据
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # 返回只包含图和标签的精简字典
        return {
            "graph": graph_data, 
            "label": label
        }

def qm9_atomistic_collate_fn(batch_list):
    """
    纯原子图 Collate 函数：
    使用 PyG 的 Batch 将独立的分子图打包在一起
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