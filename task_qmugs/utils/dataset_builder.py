# task_ED5_OE/utils/dataset_builder.py
import os

def build_dataset(data_config):
    mode = data_config.get('dataset_mode')
    pkl_path = data_config.get('pkl_path')
    targets = data_config.get('targets')

    if not targets or not isinstance(targets, list):
        raise ValueError("❌ ED5 数据集配置错误: 必须提供一个 'targets' 列表")

    # ================= 1. 电子密度场表征=================
    if mode == "repr_ed_field":
 
        from dataset_qmugs_edpointcloud import QMugsPointCloudDataset, ptv3_collate_fn
        
        grid_size = data_config.get('grid_size', 0.1)
        
        train_dataset = QMugsPointCloudDataset(
            pkl_path=pkl_path, split='train', 
            grid_size=grid_size, targets=targets
        )
        val_dataset = QMugsPointCloudDataset(
            pkl_path=pkl_path, split='valid', 
            grid_size=grid_size, targets=targets
        )
        collate_fn = ptv3_collate_fn

# ================= 2. 稠密电子密度场=================
    elif mode == "repr_ed_dense":
        from dataset_qmugs_ed_dense import QMugsDenseDataset, qmugs_dense_collate_fn
        
        grid_size = data_config.get('grid_size', 0.1)
        train_dataset = QMugsDenseDataset(pkl_path=pkl_path, split='train', grid_size=grid_size, targets=targets)
        val_dataset = QMugsDenseDataset(pkl_path=pkl_path, split='valid', grid_size=grid_size, targets=targets)
        collate_fn = qmugs_dense_collate_fn

    # ================= 3. 混合融合表征 =================
    elif mode == "repr_hybrid_fusion":
        from dataset_qmugs_hybrid_fusion import QMugsDualDataset, qmugs_dual_collate_fn
        
        grid_size = data_config.get('grid_size', 0.1)
        max_radius = data_config.get('max_radius', 5.0)
        
        train_dataset = QMugsDualDataset(
            pkl_path=pkl_path, split='train', 
            grid_size=grid_size, max_radius=max_radius, targets=targets
        )
        val_dataset = QMugsDualDataset(
            pkl_path=pkl_path, split='valid', 
            grid_size=grid_size, max_radius=max_radius, targets=targets
        )
        collate_fn = qmugs_dual_collate_fn

    # ================= 4. 纯原子图结构=================
    elif mode == "repr_atomistic_graph":
        from dataset_qmugs_atomistic_graph import QMugsAtomisticDataset, qmugs_atomistic_collate_fn
        train_dataset = QMugsAtomisticDataset(pkl_path=pkl_path, split='train', targets=targets)
        val_dataset = QMugsAtomisticDataset(pkl_path=pkl_path, split='valid', targets=targets)
        collate_fn = qmugs_atomistic_collate_fn

    elif mode == "repr_atomistic_graph_fe":
        from dataset_qmugs_atomistic_graph import QMugsAtomisticDataset, qmugs_atomistic_collate_fn
        test_dataset = QMugsAtomisticDataset(pkl_path=pkl_path, split='test', targets=targets)
        return test_dataset, qmugs_atomistic_collate_fn    
    else:
        raise ValueError(f"❌ 未知的数据集模式 (dataset_mode): {mode}")

    return train_dataset, val_dataset, collate_fn