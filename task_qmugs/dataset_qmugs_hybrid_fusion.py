import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

class QMugsDualDataset(Dataset):
    """
    QMugs 双流融合 Dataset (专为 UniFieldNet 等结合 Atom Graph 和 ED Point Cloud 的模型设计)
    """
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
        
        # ================= 动态提取标签 =================
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
        
        # ================= 1. 提取点云特征 (ED Field) =================
        # 此时坐标已确认为 Bohr 单位，直接读取
        ed_matrix = item['ed']
        ed_coords = torch.from_numpy(ed_matrix[:, :3]).float()
        ed_feat = torch.from_numpy(ed_matrix[:, 3].reshape(-1, 1)).float()
        
        # ================= 2. 提取原子图特征 (Atomistic) =================
        # 此时坐标已确认为 Bohr 单位，直接读取，不加任何乘法转换！
        mol_coords = torch.from_numpy(item['atom_coords']).float()
        atomic_numbers = torch.tensor(item['atom_types'], dtype=torch.long)
        
        # 封装为 PyG 标准 Data 对象
        graph_data = Data(pos=mol_coords, z=atomic_numbers)
        
        # ================= 3. 提取标签 =================
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # 返回统一的双流字典结构
        return {
            "graph": graph_data,
            "point_cloud": {
                "coord": ed_coords,
                "feat": ed_feat
            },
            "label": label
        }

def qmugs_dual_collate_fn(batch_list):
    """
    双流融合 Collate 函数：
    同时处理 PyG 的 disconnected subgraph 组装，以及 PTv3 的 offset 记录。
    """
    graphs = []
    pc_coords, pc_feats, offsets, batch_ids = [], [], [], []
    labels = []
    
    batch_offset = 0
    
    for i, sample in enumerate(batch_list):
        # 1. 收集 Graph (稍后统一交给 PyG 的 Batch 处理)
        graphs.append(sample["graph"])
        
        # 2. 收集 Label
        labels.append(sample["label"])
        
        # 3. 收集 Point Cloud 并维护 PTv3 需要的拓扑变量
        pc = sample["point_cloud"]
        pc_coords.append(pc["coord"])
        pc_feats.append(pc["feat"])
        
        num_points = pc["coord"].shape[0]
        
        # 记录 offset (每个样本点云在展平后的结束索引)
        batch_offset += num_points
        offsets.append(batch_offset)
        
        # 记录每个点属于当前 batch 的哪一个样本 (极其重要，防交叉)
        batch_ids.append(torch.full((num_points,), i, dtype=torch.long))
        
    # ================= 组装最终字典 =================
    # PyG 原生拼图魔法
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