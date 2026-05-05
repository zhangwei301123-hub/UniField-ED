import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

class QMugsDenseDataset(Dataset):
    """
    QMugs 稠密点云 Dataset (专为 PointNext 等需要固定点数 1024 的模型设计)
    """
    def __init__(self, pkl_path, split='train', grid_size=0.1, targets=None):
        super().__init__()
        self.split = split
        self.grid_size = grid_size
        self.targets = targets or []
        
        print(f"📂 Loading QMugs Dense Point Cloud Data from {pkl_path} [{split}]...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        # QMugs 的结构是字典划分 split，里面是一个包含多个分子字典的列表
        self.data_list = data_dict[split]
        
        # ================= 💡 核心修改：动态提取标签 =================
        # 抛弃 ED5 的 TARGET_MAP，直接通过传入的 targets 字符串去字典里拿对应的值
        all_labels = []
        for item in self.data_list:
            # 例如 item['homo_energy_meV']
            lbl = [item[t] for t in self.targets]
            all_labels.append(lbl)
            
        # 预先打包成 numpy 数组，方便 train.py 里计算 mean 和 std
        self.labels = np.array(all_labels, dtype=np.float32)
        # ============================================================
            
        print(f"✅ Loaded {len(self.data_list)} samples for targets: {self.targets}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # QMugs 的点云数据存在 'ed' 键里，形状为 (1024, 4)
        # 前 3 列是坐标 (x,y,z)，第 4 列是电子密度特征
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
    """
    QMugs 稠密点云 Collate 函数：
    将 [N, C] 的特征转置为 [C, N] 并 Stack 成内存连续的 Tensor [B, C, N]
    """
    batch_coord, batch_feat, batch_labels = [], [], []

    for data in batch:
        batch_coord.append(data['coord'])
        
        # [N, C] -> [C, N] (PointNext 核心需求)
        feat = data['feat'].transpose(0, 1)
        batch_feat.append(feat)
        
        batch_labels.append(data['label'])

    # [B, N, 3]
    pos = torch.stack(batch_coord, dim=0).contiguous()
    # [B, C, N]
    x = torch.stack(batch_feat, dim=0).contiguous()
    # [B, num_targets] (💡 修复了你原来代码里漏掉 batch_labels 的 Bug)
    labels = torch.stack(batch_labels, dim=0)

    # 完美适配通用 engine.py 
    return {
        "point_cloud": {"pos": pos, "x": x},
        "labels": labels
    }