import torch
from tqdm import tqdm

def to_device(data, device):
    """
    [核心黑科技]: 递归地将任意嵌套的数据结构移动到 GPU
    支持: Tensor, 字典, 列表, 以及 PyG 的 Data/Batch 对象
    """
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
    
    for batch_data in pbar:
        optimizer.zero_grad()

        batch_data = to_device(batch_data, device)

        labels = batch_data.get("labels", batch_data.get("label"))
        
        norm_labels = (labels - normalizer['mean']) / normalizer['std']

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

    return total_loss / len(train_loader)


def validate(model, regressor, val_loader, epoch, normalizer, config, device):
    model.eval()
    if not isinstance(regressor, torch.nn.Identity):
        regressor.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Val E{epoch}")
        for batch_data in pbar:

            batch_data = to_device(batch_data, device)
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