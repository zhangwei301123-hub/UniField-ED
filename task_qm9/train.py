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
    return parser.parse_args()

def load_and_merge_config(train_cfg_path, data_cfg_path, model_cfg_path):
    """
    读取三个 YAML (训练参数、数据模态、模型结构) 并合并
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
    
    # 加载并合并配置
    config = load_and_merge_config(args.train_config, args.data_config, args.model_config)
    
    dataset_mode = config['data'].get('dataset_mode', 'UnknownMode')
    model_name = config['model'].get('name', 'UnknownModel')
    
    targets_list = config['data'].get('targets', [])
    output_dim = len(targets_list)
    if output_dim == 0:
        raise ValueError("❌ 数据配置中找不到 targets 列表，请检查 YAML 文件！")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./logs/QM9_{dataset_mode}_{model_name}_Task{output_dim}_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True) 
    logger = setup_logger(save_dir)

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

    logger.info("Loading Data...")
    logger.info(f"Loading Dataset Mode: {dataset_mode}...")
    

    train_dataset, val_dataset, collate_fn = build_dataset(config['data'])

    num_workers = 8 if dataset_mode != "repr_atomistic_graph" else 0
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['train']['batch_size'], 
        shuffle=True, collate_fn=collate_fn, num_workers=num_workers, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config['train']['batch_size'], 
        shuffle=False, collate_fn=collate_fn, num_workers=num_workers
    )

 
    all_labels_tensor = torch.tensor(train_dataset.labels).float().to(device)
    normalizer = {
        'mean': all_labels_tensor.mean(dim=0),
        'std': all_labels_tensor.std(dim=0)
    }
    
    mean_str = str(normalizer['mean'].unsqueeze(0).cpu().numpy())
    std_str = str(normalizer['std'].unsqueeze(0).cpu().numpy())
    logger.info(f"Stats - Mean: {mean_str}, Std: {std_str}")

    logger.info(f"Building Model: {config['model']['name']}...")
    
    model, regressor = build_model(config['model'], output_dim=output_dim)
    
    model = model.to(device)
    regressor = regressor.to(device)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in regressor.parameters())
    logger.info(f"Total Parameters: {total_params / 1e6:.4f} M\n")

    # ================= 优化器 =================
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

    for epoch in range(1, config['train']['epochs'] + 1):
        # 注意：这里的 train_one_epoch 和 validate 内部可能需要兼容三种模态字典，模型通过 model(batch_data) 自行取用
        train_loss = train_one_epoch(model, regressor, train_loader, optimizer, criterion, epoch, normalizer, config, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        val_metrics, val_avg_rmse = validate(model, regressor, val_loader, epoch, normalizer, config, device)

        logger.info(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | LR: {current_lr:.6f}")
        logger.info(f"    >>> Val Avg RMSE (Real): {val_avg_rmse:.4f}")
        
        print_maes = [f'{x:.4f}' for x in val_metrics['MAE']]
        logger.info(f"    >>> Val MAE Details: {print_maes}")
        
        # 早停与保存逻辑
        if val_avg_rmse < best_rmse:
            best_rmse = val_avg_rmse
            patience_counter = 0 
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'regressor_state': regressor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'normalizer': normalizer,
                'best_rmse': best_rmse,
                'targets': targets_list # 把预测的是哪8个目标存进去，方便测试时读取
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