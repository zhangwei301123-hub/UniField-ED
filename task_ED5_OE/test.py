import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# 动态挂载项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.builder import build_model
from utils.engine import to_device  # 复用我们写好的解耦迁移利器

def setup_test_logger(log_file):
    """专门为测试脚本配置的 Logger，双向输出到终端和文件"""
    logger = logging.getLogger("TestLogger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 文件 Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # 终端 Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def build_test_dataset(data_config):
    """
    专门为测试集准备的工厂函数，动态读取 dataset_mode 并将 split 设为 'test'
    """
    mode = data_config.get('dataset_mode')
    pkl_path = data_config.get('pkl_path')
    targets = data_config.get('targets')
    grid_size = data_config.get('grid_size', 0.1)

    if mode == "repr_ed_field":
        from dataset_ed5_oe_edpointcloud import ED5OEEDPointCloudDataset, ptv3_collate_fn
        test_dataset = ED5OEEDPointCloudDataset(pkl_path=pkl_path, split='test', grid_size=grid_size, targets=targets)
        return test_dataset, ptv3_collate_fn
        
    elif mode == "repr_ed_dense":
        from dataset_ed5_oe_dense import ED5OEDenseDataset, pointnext_collate_fn
        test_dataset = ED5OEDenseDataset(pkl_path=pkl_path, split='test', grid_size=grid_size, targets=targets)
        return test_dataset, pointnext_collate_fn
        
    elif mode == "repr_hybrid_fusion":
        from dataset_ed5_oe_hybrid_fusion import ED5OEDualDataset, dual_collate_fn
        max_radius = data_config.get('max_radius', 5.0)
        test_dataset = ED5OEDualDataset(pkl_path=pkl_path, split='test', grid_size=grid_size, max_radius=max_radius, targets=targets)
        return test_dataset, dual_collate_fn

    # ================== 💡 关键补丁：添加原子图模式 ==================
    elif mode == "repr_atomistic_graph":
        from dataset_ed5_oe_atomistic_graph import ED5OEAtomisticDataset, atomistic_collate_fn
        test_dataset = ED5OEAtomisticDataset(pkl_path=pkl_path, split='test', targets=targets)
        return test_dataset, atomistic_collate_fn
    # ==============================================================
        
    else:
        raise ValueError(f"❌ 未知的数据集模式: {mode}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    # 只需要传入你训练好的模型文件夹路径即可！
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Path to the saved model directory')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================== 1. 路径与日志准备 ==================
    ckpt_path = os.path.join(args.ckpt_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ 找不到权重文件: {ckpt_path}")

    log_file = os.path.join(args.ckpt_dir, "test.log")
    logger = setup_test_logger(log_file)

    # 读取备份在文件夹里的配置文件
    train_cfg_path = os.path.join(args.ckpt_dir, 'train_config.yml')
    data_cfg_path = os.path.join(args.ckpt_dir, 'data_config.yml')
    model_cfg_path = os.path.join(args.ckpt_dir, 'model_config.yml')

    with open(data_cfg_path, 'r', encoding='utf-8') as f: data_config = yaml.safe_load(f)
    with open(model_cfg_path, 'r', encoding='utf-8') as f: model_config = yaml.safe_load(f)
    with open(train_cfg_path, 'r', encoding='utf-8') as f: train_config = yaml.safe_load(f)

    targets = data_config.get('targets', [])
    output_dim = len(targets)

    logger.info(f"🎯 Evaluating Targets: {targets}")
    logger.info(f"📂 Loading checkpoint from: {ckpt_path}")
    logger.info(f"📝 Test log will be saved to: {log_file}")

    # ================== 2. 加载数据 ==================
    test_dataset, collate_fn = build_test_dataset(data_config)
    num_workers = 8 if data_config.get('dataset_mode') != "repr_atomistic_graph" else 0
    
    test_loader = DataLoader(
        test_dataset, batch_size=train_config.get('batch_size', 32), 
        shuffle=False, collate_fn=collate_fn, num_workers=num_workers
    )

    # ================== 3. 构建模型与恢复权重 ==================
    model, regressor = build_model(model_config, output_dim=output_dim)
    model = model.to(device)
    regressor = regressor.to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    if not isinstance(regressor, nn.Identity):
        regressor.load_state_dict(checkpoint['regressor_state'])
    
    # 提取存下来的均值和方差，用于反归一化
    normalizer = checkpoint['normalizer']
    best_epoch = checkpoint.get('epoch', 'Unknown')
    val_rmse = checkpoint.get('best_rmse', 'Unknown')

    logger.info(f"✅ Successfully loaded checkpoint from Epoch {best_epoch}")
    if isinstance(val_rmse, float):
        logger.info(f"    Validation Avg RMSE at this epoch was: {val_rmse:.4f}")

    # ================== 4. 测试循环 ==================
    model.eval()
    if not isinstance(regressor, nn.Identity):
        regressor.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch_data in pbar:
            batch_data = to_device(batch_data, device)
            labels = batch_data.get("labels", batch_data.get("label"))


            if "point_cloud" in batch_data:
                batch_data.update(batch_data["point_cloud"])
            
            if "grid_size" not in batch_data:
                batch_data["grid_size"] = data_config.get("grid_size", 0.1)
            # ========================================================

            feat_out = model(batch_data)
# ================== 💡 输出格式兼容补丁 (增强版) ==================
            if hasattr(feat_out, 'feat'):
                batch_idx = batch_data.get('batch', None)
                
                # 如果老数据集没有提供 batch，但提供了 offset，我们当场还原 batch 索引
                if batch_idx is None and 'offset' in batch_data:
                    offset = batch_data['offset']
                    batch_idx = torch.zeros(feat_out.feat.shape[0], dtype=torch.long, device=device)
                    start = 0
                    for b_id, end in enumerate(offset):
                        batch_idx[start:end] = b_id
                        start = end

                # 现在一定有 batch_idx 了，执行分子级全局池化
                if batch_idx is not None:
                    from torch_scatter import scatter
                    feat_out = scatter(feat_out.feat, batch_idx, dim=0, reduce='mean')
                else:
                    feat_out = feat_out.feat
            # ========================================================
            # ========================================================

            pred_norm = regressor(feat_out)

            # 反归一化到真实尺度 (确保 Tensor 形状对齐)
            std = normalizer['std'].to(device)
            mean = normalizer['mean'].to(device)
            pred_real = pred_norm * std + mean

            all_preds.append(pred_real.cpu().numpy())
            all_labels.append(labels.cpu().numpy())


    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # ================== 5. 指标计算与日志打印 ==================

    logger.info("========================================")
    logger.info(" 🎉 Test Set Evaluation Results 🎉")
    logger.info("========================================")

    all_rmses = [] # [💡 新增] 用来收集所有属性的 RMSE

    # 因为是多任务，我们遍历每一个 Target 进行打印
    for i, target_name in enumerate(targets):
        pred_i = all_preds[:, i]
        true_i = all_labels[:, i]

        rmse = np.sqrt(np.mean((pred_i - true_i) ** 2))
        mae = np.mean(np.abs(pred_i - true_i))
        
        all_rmses.append(rmse) # [💡 新增] 将当前属性的 RMSE 存入列表
        
        # 防止常数预测导致 Pearson 计算报错
        if np.std(pred_i) > 1e-6 and np.std(true_i) > 1e-6:
            pearson = pearsonr(pred_i, true_i)[0]
            spearman = spearmanr(pred_i, true_i)[0]
        else:
            pearson, spearman = 0.0, 0.0

        logger.info(f"Target Property : {target_name}")
        logger.info(f"Test RMSE       : {rmse:.4f}")
        logger.info(f"Test MAE        : {mae:.4f}")
        logger.info(f"Test Pearson r  : {pearson:.4f}")
        logger.info(f"Test Spearman r : {spearman:.4f}")
        logger.info("----------------------------------------")

    # [💡 新增] 计算并打印全局平均 RMSE
    avg_rmse = np.mean(all_rmses)
    logger.info(f"🏆 Average Test RMSE (All {len(targets)} Targets): {avg_rmse:.4f}")
    logger.info("========================================")

if __name__ == "__main__":
    main()