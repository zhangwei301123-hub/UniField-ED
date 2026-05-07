import torch
from tqdm import tqdm

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif hasattr(data, 'to'): 
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    return data

def train_one_epoch(model, regressor, train_loader, optimizer, criterion, epoch, normalizer, config, device):
    model.train()
    if not isinstance(regressor, torch.nn.Identity):
        regressor.train()

    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Train E{epoch}")
    default_grid_size = config['data'].get("grid_size", 0.1)
    
    for batch_data in pbar:
        optimizer.zero_grad()
        batch_data = to_device(batch_data, device)


        if "point_cloud" in batch_data:
            if "grid_size" not in batch_data["point_cloud"]:
                batch_data["point_cloud"]["grid_size"] = default_grid_size
        elif "grid_size" not in batch_data:
            batch_data["grid_size"] = default_grid_size

        labels = batch_data.get("labels", batch_data.get("label"))
        norm_labels = (labels - normalizer['mean']) / normalizer['std']

    
        try:
            feat_out = model(batch_data)
            pred = regressor(feat_out)
            
            loss = criterion(pred, norm_labels)
            loss.backward()

            if config['train'].get('grad_norm_clip'):
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(regressor.parameters()), 
                    config['train']['grad_norm_clip']
                )

            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        except RuntimeError as e:
          
            if "must match the existing size" in str(e):
                print(f"\n⚠️ [Epoch {epoch}] 跳过了一个由于 DIG 库索引 Bug 导致的坏 Batch (Size: {labels.size(0)})")
                optimizer.zero_grad() # 确保梯度清零，不污染下一批数据
                continue
            else:
                
                raise e
   

    return total_loss / len(train_loader) if len(train_loader) > 0 else 0

def validate(model, regressor, val_loader, epoch, normalizer, config, device):
    model.eval()
    if not isinstance(regressor, torch.nn.Identity):
        regressor.eval()

    all_preds = []
    all_labels = []
    default_grid_size = config['data'].get("grid_size", 0.1)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Val E{epoch}")
        for batch_data in pbar:
            batch_data = to_device(batch_data, device)
            
            if "point_cloud" in batch_data:
                if "grid_size" not in batch_data["point_cloud"]:
                    batch_data["point_cloud"]["grid_size"] = default_grid_size
            elif "grid_size" not in batch_data:
                batch_data["grid_size"] = default_grid_size

            labels = batch_data.get("labels", batch_data.get("label"))
            feat_out = model(batch_data)
            pred_norm = regressor(feat_out)

            pred_real = pred_norm * normalizer['std'] + normalizer['mean']
            all_preds.append(pred_real)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    mse = torch.mean((all_preds - all_labels) ** 2).item()
    rmse = mse ** 0.5
    mae_per_target = torch.mean(torch.abs(all_preds - all_labels), dim=0).cpu().numpy()

    metrics = {'MAE': mae_per_target.tolist()}
    return metrics, rmse