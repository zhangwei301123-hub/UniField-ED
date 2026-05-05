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

# [💡 ED5特有修改] 动态挂载项目根目录，以便安全导入共享区 models
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入共享工具和当前 task_ed5 的专属模块
from utils.builder import build_model
from utils.dataset_builder import build_dataset
from utils.logger import setup_logger
from utils.engine import train_one_epoch, validate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1', help='GPU ID to use')
    # [💡 ED5特有修改] 去掉了 --target 参数，目标现在由 data_config 内的 targets 列表决定
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
    
    # [💡 ED5特有修改] 获取多任务列表，推导输出维度
    targets_list = config['data'].get('targets', [])
    output_dim = len(targets_list)
    if output_dim == 0:
        raise ValueError("❌ 数据配置中找不到 targets 列表，请检查 YAML 文件！")
    
    # 创建保存目录 (命名格式: ED5_repr_ed_field_PTv3_MultiTask8_2026xxxx)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./logs/ED5_{dataset_mode}_{model_name}_MultiTask{output_dim}_{timestamp}"
    logger = setup_logger(save_dir)

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
    
    # [💡 ED5特有修改] 直接传入 data_config，不需要传 args.target
    train_dataset, val_dataset, collate_fn = build_dataset(config['data'])

    # 如果是分子图 (AtomOnly)，可能不支持 num_workers 多线程，这里做个安全兼容
    num_workers = 8 if dataset_mode != "repr_atomistic_graph" else 0
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['train']['batch_size'], 
        shuffle=True, collate_fn=collate_fn, num_workers=num_workers, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config['train']['batch_size'], 
        shuffle=False, collate_fn=collate_fn, num_workers=num_workers
    )

    # ================= 计算统计量 =================
    # [💡 ED5特有修改] ED5 数据集的属性是 labels，并且本身就是 (N, 8) 形状，不需要 view(-1, 1) 打平
    all_labels_tensor = torch.tensor(train_dataset.labels).float().to(device)
    normalizer = {
        'mean': all_labels_tensor.mean(dim=0),
        'std': all_labels_tensor.std(dim=0)
    }
    
    # [💡 格式修改] 使用 unsqueeze(0) 将 (N,) 变成 (1, N)，转换为 numpy 后自带 [[ ]] 格式同行输出
    mean_str = str(normalizer['mean'].unsqueeze(0).cpu().numpy())
    std_str = str(normalizer['std'].unsqueeze(0).cpu().numpy())
    logger.info(f"Stats - Mean: {mean_str}, Std: {std_str}")

    # ================= 模型构建 =================
    logger.info(f"Building Model: {config['model']['name']}...")
    
    # [💡 ED5特有修改] 必须将推导出的 output_dim (例如 8) 传给 builder！
    model, regressor = build_model(config['model'], output_dim=output_dim)
    
    model = model.to(device)
    regressor = regressor.to(device)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in regressor.parameters())
    logger.info(f"Total Parameters: {total_params / 1e6:.4f} M\n")

    # ================= 优化器 =================
    # [💡 ED5特有修改] 多任务联合预测，MSELoss 会自动对所有 8 个属性的损失求平均
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

        # [💡 格式修改] 去掉了 \n，防止时间戳前缀换行断层
        logger.info(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | LR: {current_lr:.6f}")
        logger.info(f"    >>> Val Avg RMSE (Real): {val_avg_rmse:.4f}")
        
        # [💡 格式修改] 格式化所有的 MAE 值并作为完整列表打印
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