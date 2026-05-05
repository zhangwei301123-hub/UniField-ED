import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import pprint
import shutil

# 动态挂载项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.builder import build_model
from utils.dataset_builder import build_dataset
from utils.logger import setup_logger
from utils.engine import train_one_epoch, validate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1', help='GPU ID to use')
    parser.add_argument('--train_config', type=str, default='./configs/train_base.yml')
    parser.add_argument('--data_config', type=str, default='./configs/data/repr_ed_field.yml')
    parser.add_argument('--model_config', type=str, default='./configs/models/PTv3.yml')

    # 初始化权重，但不恢复 optimizer / scheduler / epoch
    parser.add_argument(
        '--init_ckpt',
        type=str,
        default=None,
        help='Path to a best checkpoint (.pth). Only model/regressor weights will be loaded as initialization.'
    )
    return parser.parse_args()


def load_and_merge_config(train_cfg_path, data_cfg_path, model_cfg_path):
    """
    读取三个 YAML 并合并
    """
    with open(train_cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    with open(data_cfg_path, 'r', encoding='utf-8') as f:
        config['data'] = yaml.safe_load(f)

    with open(model_cfg_path, 'r', encoding='utf-8') as f:
        config['model'] = yaml.safe_load(f)

    return config


def main():
    args = parse_args()

    # 环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载配置
    config = load_and_merge_config(args.train_config, args.data_config, args.model_config)

    dataset_mode = config['data'].get('dataset_mode', 'UnknownMode')
    model_name = config['model'].get('name', 'UnknownModel')
    targets_list = config['data'].get('targets', [])
    output_dim = len(targets_list)

    if output_dim == 0:
        raise ValueError("❌ 数据配置中找不到 targets 列表，请检查 YAML 文件！")

    # 新实验目录：每次都新建
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    init_tag = "scratch" if args.init_ckpt is None else "init_from_ckpt"
    save_dir = f"./logs/ED5OE_{dataset_mode}_{model_name}_MultiTask{output_dim}_{init_tag}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(save_dir)

    # 备份配置
    shutil.copy(args.train_config, os.path.join(save_dir, 'train_config.yml'))
    shutil.copy(args.data_config, os.path.join(save_dir, 'data_config.yml'))
    shutil.copy(args.model_config, os.path.join(save_dir, 'model_config.yml'))
    logger.info(f"💾 Config files backed up to {save_dir}")

    logger.info(f"🎯 Multi-Task Targets ({output_dim}): {targets_list}")
    logger.info("=" * 40)
    logger.info("        Training Configuration        ")
    logger.info("=" * 40)
    logger.info("\n" + pprint.pformat(config))
    logger.info("=" * 40 + "\n")
    logger.info(f"🚀 Start Training on Device: {device}")

    # ================= 数据加载 =================
    logger.info("Loading Data...")
    logger.info(f"Loading Dataset Mode: {dataset_mode}...")

    train_dataset, val_dataset, collate_fn = build_dataset(config['data'])

    num_workers = 8 if dataset_mode != "repr_atomistic_graph" else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    # ================= 计算统计量 =================
    all_labels_tensor = torch.tensor(train_dataset.labels).float().to(device)
    normalizer = {
        'mean': all_labels_tensor.mean(dim=0),
        'std': all_labels_tensor.std(dim=0)
    }

    mean_str = str(normalizer['mean'].unsqueeze(0).cpu().numpy())
    std_str = str(normalizer['std'].unsqueeze(0).cpu().numpy())
    logger.info(f"Stats - Mean: {mean_str}, Std: {std_str}")

    # ================= 模型构建 =================
    logger.info(f"Building Model: {config['model']['name']}...")
    model, regressor = build_model(config['model'], output_dim=output_dim, normalizer=normalizer)

    model = model.to(device)
    regressor = regressor.to(device)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in regressor.parameters())
    logger.info(f"Total Parameters: {total_params / 1e6:.4f} M\n")

    # ================= 初始化权重（只加载模型，不恢复训练状态） =================
    if args.init_ckpt is not None:
        if not os.path.exists(args.init_ckpt):
            raise FileNotFoundError(f"❌ 找不到初始化 checkpoint: {args.init_ckpt}")

        logger.info("🧩 Loading initialization checkpoint...")
        checkpoint = torch.load(args.init_ckpt, map_location=device)

        if 'model_state' not in checkpoint or 'regressor_state' not in checkpoint:
            raise KeyError("❌ checkpoint 中缺少 model_state 或 regressor_state，无法初始化模型。")

        model.load_state_dict(checkpoint['model_state'])
        regressor.load_state_dict(checkpoint['regressor_state'])

        logger.info(f"✅ Loaded weights from: {args.init_ckpt}")
        logger.info("⚠️ 仅加载模型参数，optimizer / scheduler / epoch 不恢复，训练将从 epoch 0 重新开始。")

    # ================= 优化器与调度器（重新初始化） =================
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(model.parameters()) + list(regressor.parameters()),
        lr=float(config['train']['lr']),
        weight_decay=float(config['train']['weight_decay'])
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])

    # ================= 训练循环 =================
    best_rmse = float('inf')
    patience_counter = 0

    # 从 epoch 0 开始
    for epoch in range(0, config['train']['epochs']):
        train_loss = train_one_epoch(
            model, regressor, train_loader, optimizer, criterion, epoch, normalizer, config, device
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        val_metrics, val_avg_rmse = validate(
            model, regressor, val_loader, epoch, normalizer, config, device
        )

        logger.info(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | LR: {current_lr:.6f}")
        logger.info(f"    >>> Val Avg RMSE (Real): {val_avg_rmse:.4f}")

        print_maes = [f'{x:.4f}' for x in val_metrics['MAE']]
        logger.info(f"    >>> Val MAE Details: {print_maes}")

        # 保存最优模型
        if val_avg_rmse < best_rmse:
            best_rmse = val_avg_rmse
            patience_counter = 0
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'regressor_state': regressor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'normalizer': normalizer,
                'best_rmse': best_rmse,
                'targets': targets_list,
                'init_ckpt': args.init_ckpt
            }, save_path)
            logger.info(f"💾 Saved Best Model to {save_path}")
        else:
            patience_counter += 1
            logger.info(f"⏳ No improvement. Patience: {patience_counter}/{config['train']['patience']}")
            if patience_counter >= config['train']['patience']:
                logger.info(f"🛑 Early Stopping triggered! Best RMSE: {best_rmse:.4f}")
                break


if __name__ == "__main__":
    main()