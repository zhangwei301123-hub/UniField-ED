import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

class QM9EDPointCloudDataset(Dataset):
    """
    支持动态选择 target 属性的 QM9 点云数据集
    """
    def __init__(self, pkl_path, split='train', grid_size=0.1, target='U0_meV'):
        super().__init__()
        self.split = split
        self.grid_size = grid_size
        self.target = target

        print(f"📂 Loading {split} data from {pkl_path}")
        print(f"🎯 Target Property: {target}")

        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)

        split_data = data_dict[split]

        self.ed_list = []
        self.label_list = []

        for mol_id, val in split_data.items():
            # 点云
            self.ed_list.append(val['ed'])

            # 检查 target 是否存在
            if target not in val:
                raise KeyError(
                    f"❌ Target '{target}' not found!\n"
                    f"Available keys: {list(val.keys())}"
                )

            self.label_list.append(val[target])

        print(f"✅ Loaded {len(self.ed_list)} samples for [{split}]")

    def __len__(self):
        return len(self.ed_list)

    def __getitem__(self, idx):
        raw_ed = self.ed_list[idx]

        coord = raw_ed[:, :3]
        feat = raw_ed[:, 3].reshape(-1, 1)

        label = np.array([self.label_list[idx]], dtype=np.float32)

        return {
            "coord": torch.from_numpy(coord).float(),
            "feat": torch.from_numpy(feat).float(),
            "label": torch.from_numpy(label).float()
        }


def ptv3_collate_fn(batch):
    coords = []
    feats = []
    labels = []
    offsets = []
    batch_offset = 0

    for sample in batch:
        coords.append(sample["coord"])
        feats.append(sample["feat"])
        labels.append(sample["label"])

        batch_offset += sample["coord"].shape[0]
        offsets.append(batch_offset)

    coord_batch = torch.cat(coords, dim=0)
    feat_batch = torch.cat(feats, dim=0)
    label_batch = torch.stack(labels, dim=0)
    offset_batch = torch.tensor(offsets, dtype=torch.long)

    return {
        "coord": coord_batch,
        "feat": feat_batch,
        "offset": offset_batch,
        "labels": label_batch
    }