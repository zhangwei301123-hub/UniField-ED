# utils/dataset_builder.py


def build_dataset(data_config, target):
    """
    根据配置文件动态构建 Dataset 和 collate_fn
    """
    dataset_name = data_config.get('name', 'QM9EDPointCloud')

    # ================= 1. QM9 数据集 =================
    if dataset_name == "QM9EDPointCloud":
        from datasets.dataset_qm9_edpointcloud import QM9EDPointCloudDataset, ptv3_collate_fn
        
        valid_targets = [
            'mu_mD', 'alpha_mBohr3', 'homo_meV', 'lumo_meV', 'gap_meV',
            'r2_mBohr2', 'zpve_meV', 'U0_meV', 'U_meV', 'H_meV', 'G_meV', 'Cv_mcal'
        ]
        if target not in valid_targets:
            raise ValueError(f"❌ Invalid target '{target}' for dataset {dataset_name}.")

        train_dataset = QM9EDPointCloudDataset(
            data_config['pkl_path'], split='train', 
            grid_size=data_config['grid_size'], target=target
        )
        val_dataset = QM9EDPointCloudDataset(
            data_config['pkl_path'], split='valid', 
            grid_size=data_config['grid_size'], target=target
        )
        collate_fn = ptv3_collate_fn
# ================= 2. ED5_OE 数据集 =================
    elif dataset_name == "ED5OEPointCloud":  
        
        from datasets.dataset_ed5_oe_edpointcloud import ED5OEEDPointCloudDataset,  ptv3_collate_fn
        
        # ED5_OE 可能只有部分属性可以预测
        valid_targets = ['homo_meV', 'lumo_meV'] 
        if target not in valid_targets:
            raise ValueError(f"❌ Invalid target '{target}' for dataset {dataset_name}.")

        train_dataset = ED5OEEDPointCloudDataset(
            data_config['pkl_path'], split='train', 
            grid_size=data_config['grid_size'], target=target
        )
        val_dataset = ED5OEEDPointCloudDataset(
            data_config['pkl_path'], split='valid', 
            grid_size=data_config['grid_size'], target=target
        )
        collate_fn = ptv3_collate_fn

    else:
        raise ValueError(f"❌ Unsupported dataset: {dataset_name}")

    return train_dataset, val_dataset, collate_fn