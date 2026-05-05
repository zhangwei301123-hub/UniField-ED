import numpy as np
from scipy.stats import pearsonr, spearmanr

def metric_reg_multitask(logits, targets):
    """
    计算回归指标: MAE, RMSE, Pearson, Spearman
    """
    num_tasks = logits.shape[1]
    mae_list, rmse_list, pearson_list, spearman_list = [], [], [], []

    for i in range(num_tasks):
        pred = logits[:, i]
        true = targets[:, i]

        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))

        if np.std(pred) < 1e-6 or np.std(true) < 1e-6:
            pearson, spearman = 0.0, 0.0
        else:
            pearson = pearsonr(pred, true)[0]
            spearman = spearmanr(pred, true)[0]

        mae_list.append(mae)
        rmse_list.append(rmse)
        pearson_list.append(pearson)
        spearman_list.append(spearman)

    metrics = {
        "MAE": mae_list,
        "RMSE": rmse_list,
        "Pearson": pearson_list,
        "Spearman": spearman_list
    }
    return metrics, np.mean(rmse_list)