import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
from tqdm import tqdm
import os
import sys
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

# 确保路径正确
current_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from models.model import MSGA_RNA


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    return {
        'seed': 42,
        'batch_size': 16,
        'lr': 1e-4,
        'weight_decay': 0.1,
        'embed_dim': 128,
        'nhead': 1,
        'fusion_hidden_dim': 64,
        'dropout': 0.1,
        'aux_weight_macro': 0.4,
        'aux_weight_micro': 0.5,
        'scheduler_factor': 0.1,
        'scheduler_patience': 40,
        'early_stop_patience': 60,
        'epochs': 150,
        'n_output': 1,
        'data_path': r'data/processed/final_data_random.pt'
    }


class EarlyStopping:
    """
    根据验证集 Val RMSE 做早停，并保存 Val RMSE 最优的模型。

    Train：用于训练模型。
    Val：用于早停、调学习率、选择最佳模型。
    Test：不参与训练和模型选择，只在最后评估一次。
    """

    def __init__(self, patience=60, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_rmse_min = np.inf
        self.best_metrics = {}
        self.best_model_path = None

    def __call__(self, val_rmse, model, model_full_path, current_metrics, fold=None):
        score = -val_rmse

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_rmse, model, model_full_path, current_metrics, fold)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_rmse, model, model_full_path, current_metrics, fold)
            self.counter = 0

    def save_checkpoint(self, val_rmse, model, model_full_path, current_metrics, fold):
        checkpoint_dir = os.path.join('checkpoints', os.path.dirname(model_full_path))
        os.makedirs(checkpoint_dir, exist_ok=True)

        suffix = f"_fold{fold}" if fold is not None else ""
        save_path = os.path.join('checkpoints', f"{model_full_path}{suffix}.pt")

        torch.save(model.state_dict(), save_path)

        self.val_rmse_min = val_rmse
        self.best_metrics = current_metrics
        self.best_model_path = save_path


class TripleTowerDataset(Dataset):
    def __init__(self, data_list):
        self.raw_data = data_list

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]
        return {
            'rna_orig': item['rna_orig'].half(),
            'rna_window': item['rna_window'].half(),
            'mol_global': item['mol_global'].half(),
            'mol_frags': item['mol_frags'].half(),
            'label': item['label'].to(torch.float32)
        }


def collate_fn(batch):
    rna_o = pad_sequence([item['rna_orig'] for item in batch], batch_first=True)
    rna_w = pad_sequence([item['rna_window'] for item in batch], batch_first=True)
    frags = pad_sequence([item['mol_frags'] for item in batch], batch_first=True)

    return {
        'rna_o': rna_o,
        'rna_o_mask': (rna_o.sum(dim=-1) == 0),

        'rna_w': rna_w,
        'rna_w_mask': (rna_w.sum(dim=-1) == 0),

        'frag_embeds': frags,
        'frag_mask': (frags.sum(dim=-1) == 0),

        'global_mol_embed': torch.stack([item['mol_global'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch])
    }


def calculate_metrics(labels, preds):
    labels = np.asarray(labels).ravel()
    preds = np.asarray(preds).ravel()

    mae = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    r2 = r2_score(labels, preds)

    pcc, scc = 0.0, 0.0
    if len(np.unique(labels)) > 1 and len(np.unique(preds)) > 1:
        pcc, _ = pearsonr(labels, preds)
        scc, _ = spearmanr(labels, preds)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "PCC": float(pcc),
        "SCC": float(scc)
    }


def forward_batch(model, batch, device, use_amp=True):
    """
    统一模型前向传播，避免 Train / Val / Test 三处重复写输入。
    """

    amp_enabled = (device.type == 'cuda') and use_amp

    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
        res = model(
            rna_orig=batch['rna_o'].to(device, dtype=torch.float32),
            rna_window=batch['rna_w'].to(device, dtype=torch.float32),
            mol_global=batch['global_mol_embed'].to(device, dtype=torch.float32),
            mol_frags=batch['frag_embeds'].to(device, dtype=torch.float32),

            rna_orig_mask=batch['rna_o_mask'].to(device),
            rna_window_mask=batch['rna_w_mask'].to(device),
            frag_mask=batch['frag_mask'].to(device)
        )

    return res


def evaluate_model(model, data_loader, device, criterion=None, split_name="Test"):
    """
    用于 Val 或 Test 的评估。

    Val：
        每个 epoch 后评估，用于早停和保存最佳模型。

    Test：
        只在最佳模型确定后评估一次，不参与训练、调参、早停。
    """

    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            labels = batch['labels'].to(device).view(-1, 1)

            res = forward_batch(model, batch, device, use_amp=True)
            preds = res['out']

            if criterion is not None:
                loss = criterion(preds, labels)
                total_loss += loss.item()

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(batch['labels'].detach().cpu().numpy())

    if len(all_preds) == 0:
        raise ValueError(f"{split_name} loader 为空，无法评估。请检查该折的索引数量。")

    metrics = calculate_metrics(
        np.concatenate(all_labels).ravel(),
        np.concatenate(all_preds).ravel()
    )

    if criterion is not None:
        metrics[f"{split_name}_Loss"] = float(total_loss / max(len(data_loader), 1))

    return metrics


def train_session(train_loader, val_loader, args, device, model_full_path, fold_idx=None):
    """
    单折训练流程。

    Train：
        用于反向传播和参数更新。

    Val：
        用于学习率调度、早停、保存最佳模型。

    返回：
        best_val_metrics：Val 最优指标
        history：训练历史
        best_model_path：Val 最优模型路径
    """

    criterion = nn.SmoothL1Loss()

    model = MSGA_RNA(args).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args['lr'],
        weight_decay=args.get('weight_decay', 0.1)
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.get('scheduler_factor', 0.5),
        patience=args.get('scheduler_patience', 25)
    )

    early_stopping = EarlyStopping(
        patience=args.get('early_stop_patience', 60)
    )

    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=(device.type == 'cuda')
    )

    history = {
        "train_loss": [],
        "train_mae": [],

        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_pcc": [],
        "val_scc": [],
        "val_r2": []
    }

    total_epochs = args.get('epochs', 200)

    for epoch in range(total_epochs):
        model.train()

        total_train_loss = 0.0
        total_train_mae = 0.0

        for batch in tqdm(
            train_loader,
            desc=f"Train | Fold {fold_idx} | Ep {epoch + 1}",
            leave=False
        ):
            optimizer.zero_grad()

            labels = batch['labels'].to(device).view(-1, 1)

            res = forward_batch(model, batch, device, use_amp=True)

            loss_main = criterion(res['out'], labels)
            loss_aux_macro = criterion(res['out_macro'], labels)
            loss_aux_micro = criterion(res['out_micro'], labels)

            w_macro = args.get('aux_weight_macro', 0.0)
            w_micro = args.get('aux_weight_micro', 0.0)

            loss = loss_main + (w_macro * loss_aux_macro) + (w_micro * loss_aux_micro)

            with torch.no_grad():
                batch_mae = torch.abs(res['out'].detach() - labels).mean()
                total_train_mae += batch_mae.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        avg_train_mae = total_train_mae / max(len(train_loader), 1)

        # Val 验证：只用于模型选择，不作为最终 Test 结果
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            criterion=criterion,
            split_name="val"
        )

        avg_val_loss = val_metrics["val_Loss"]

        history["train_loss"].append(float(avg_train_loss))
        history["train_mae"].append(float(avg_train_mae))

        history["val_loss"].append(float(avg_val_loss))
        history["val_mae"].append(float(val_metrics["MAE"]))
        history["val_rmse"].append(float(val_metrics["RMSE"]))
        history["val_pcc"].append(float(val_metrics["PCC"]))
        history["val_scc"].append(float(val_metrics["SCC"]))
        history["val_r2"].append(float(val_metrics["R2"]))

        scheduler.step(val_metrics['RMSE'])

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train MAE: {avg_train_mae:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val MAE: {val_metrics['MAE']:.4f} | "
            f"Val RMSE: {val_metrics['RMSE']:.4f} | "
            f"Val PCC: {val_metrics['PCC']:.4f} | "
            f"Val SCC: {val_metrics['SCC']:.4f} | "
            f"Val R2: {val_metrics['R2']:.4f}"
        )

        # 只用 Val RMSE 早停和保存最佳模型
        early_stopping(
            val_metrics['RMSE'],
            model,
            model_full_path,
            val_metrics,
            fold=fold_idx
        )

        if early_stopping.early_stop:
            print(
                f"⏹️ Fold {fold_idx} 触发早停："
                f"连续 {early_stopping.patience} 个 epoch Val RMSE 未提升。"
            )
            break

    return early_stopping.best_metrics, history, early_stopping.best_model_path


def TraValTes(args):
    """
    五折 Train / Val / Test 流程。

    每一折：
        1. 使用 train_idx 训练模型；
        2. 使用 val_idx 早停并保存最佳模型；
        3. 加载 Val 最优模型；
        4. 使用 test_idx 做最终独立测试；
        5. 保存每折 Test 指标；
        6. 汇总五折 Test 平均结果。
    """

    seed_everything(args.get('seed', 42))

    data_pack = torch.load(args['data_path'], weights_only=True)
    raw_samples = data_pack['dataset'] if isinstance(data_pack, dict) and 'dataset' in data_pack else data_pack

    if not isinstance(data_pack, dict) or 'folds' not in data_pack:
        print("❌ 数据文件中没有 folds，无法执行 Train/Val/Test 五折训练。")
        return None

    # 动态探测输入维度
    sample = raw_samples[0]
    args['rinalmo_dim'] = sample['rna_orig'].shape[-1]
    args['chemberta_dim'] = sample['mol_global'].shape[-1]
    args['melm_dim'] = sample['mol_frags'].shape[-1]

    if args['embed_dim'] % args['nhead'] != 0:
        print("❌ embed_dim 必须能被 nhead 整除。")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 当前设备: {device}")

    full_dataset = TripleTowerDataset(raw_samples)

    dataset_tag = os.path.basename(args['data_path']).replace('.pt', '')
    exp_dim = args.get('exp_dimension', 'final_run')

    all_fold_val_metrics = []
    all_fold_test_metrics = []
    all_fold_records = []

    base_name = None

    for fold_data in data_pack['folds']:
        f_idx = fold_data['fold']

        train_idx = fold_data['train_idx']
        val_idx = fold_data['val_idx']
        test_idx = fold_data['test_idx']

        print(
            f"\n========== Fold {f_idx} =========="
            f"\nTrain: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}"
        )

        train_loader = DataLoader(
            Subset(full_dataset, train_idx),
            batch_size=args['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )

        val_loader = DataLoader(
            Subset(full_dataset, val_idx),
            batch_size=args['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )

        test_loader = DataLoader(
            Subset(full_dataset, test_idx),
            batch_size=args['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )

        base_name = (
            f"w{args.get('rna_w', 1)}_"
            f"max{args.get('mol_max', 10)}_"
            f"bs{args.get('batch_size', 8)}_"
            f"lr{args.get('lr', 1e-5)}_"
            f"dim{args.get('embed_dim', 128)}_"
            f"ly{args.get('mol_encoder_layers', 2)}_"
            f"cal{args.get('cross_align_layers', 1)}_"
            f"hd{args.get('nhead', 1)}_"
            f"ffn{args.get('fusion_hidden_dim', 128)}_"
            f"drop{args.get('dropout', 0.4)}_"
            f"aMac{args.get('aux_weight_macro', 0.3)}_"
            f"aMic{args.get('aux_weight_micro', 0.5)}_"
            f"fac{args.get('scheduler_factor', 0.7)}_"
            f"pLR{args.get('scheduler_patience', 40)}_"
            f"stop{args.get('early_stop_patience', 60)}_"
            f"ep{args.get('epochs', 200)}"
        )

        model_full_path = os.path.join(dataset_tag, exp_dim, base_name)

        # 1. Train 训练 + Val 选择最佳模型
        best_val_metrics, history, best_model_path = train_session(
            train_loader,
            val_loader,
            args,
            device,
            model_full_path,
            fold_idx=f_idx
        )

        all_fold_val_metrics.append(best_val_metrics)

        # 2. 加载 Val 最优模型
        best_model = MSGA_RNA(args).to(device)
        best_model.load_state_dict(
            torch.load(best_model_path, map_location=device)
        )

        # 3. Test 独立测试：只评估一次，不参与训练和模型选择
        criterion = nn.SmoothL1Loss()
        test_metrics = evaluate_model(
            best_model,
            test_loader,
            device,
            criterion=criterion,
            split_name="test"
        )

        all_fold_test_metrics.append(test_metrics)

        print(
            f"✅ Fold {f_idx} Test | "
            f"MAE: {test_metrics['MAE']:.4f} | "
            f"RMSE: {test_metrics['RMSE']:.4f} | "
            f"R2: {test_metrics['R2']:.4f} | "
            f"PCC: {test_metrics['PCC']:.4f} | "
            f"SCC: {test_metrics['SCC']:.4f}"
        )

        # 保存每折训练历史
        log_dir = os.path.join('logs', dataset_tag, exp_dim)
        os.makedirs(log_dir, exist_ok=True)

        with open(
            os.path.join(log_dir, f"{base_name}_fold{f_idx}_history.json"),
            'w',
            encoding='utf-8'
        ) as f_obj:
            json.dump(history, f_obj, ensure_ascii=False, indent=2)

        # 保存每折 Val 最优指标 + Test 指标
        fold_record = {
            "Fold": f_idx,

            "Train_Size": len(train_idx),
            "Val_Size": len(val_idx),
            "Test_Size": len(test_idx),

            "Best_Val_MAE": best_val_metrics["MAE"],
            "Best_Val_RMSE": best_val_metrics["RMSE"],
            "Best_Val_R2": best_val_metrics["R2"],
            "Best_Val_PCC": best_val_metrics["PCC"],
            "Best_Val_SCC": best_val_metrics["SCC"],

            "Test_MAE": test_metrics["MAE"],
            "Test_RMSE": test_metrics["RMSE"],
            "Test_R2": test_metrics["R2"],
            "Test_PCC": test_metrics["PCC"],
            "Test_SCC": test_metrics["SCC"],

            "Best_Model_Path": best_model_path
        }

        all_fold_records.append(fold_record)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. 五折平均结果：同时计算均值和标准差
    def metric_mean_std(metric_list, metric_name):
        values = [x[metric_name] for x in metric_list]
        return float(np.mean(values)), float(np.std(values))

    avg_test_rmse, std_test_rmse = metric_mean_std(all_fold_test_metrics, 'RMSE')
    avg_test_mae, std_test_mae = metric_mean_std(all_fold_test_metrics, 'MAE')
    avg_test_r2, std_test_r2 = metric_mean_std(all_fold_test_metrics, 'R2')
    avg_test_pcc, std_test_pcc = metric_mean_std(all_fold_test_metrics, 'PCC')
    avg_test_scc, std_test_scc = metric_mean_std(all_fold_test_metrics, 'SCC')

    avg_val_rmse, std_val_rmse = metric_mean_std(all_fold_val_metrics, 'RMSE')
    avg_val_mae, std_val_mae = metric_mean_std(all_fold_val_metrics, 'MAE')
    avg_val_r2, std_val_r2 = metric_mean_std(all_fold_val_metrics, 'R2')
    avg_val_pcc, std_val_pcc = metric_mean_std(all_fold_val_metrics, 'PCC')
    avg_val_scc, std_val_scc = metric_mean_std(all_fold_val_metrics, 'SCC')

    avg_results = {
        # 这 5 个字段保留原来的名字，兼容 auto_tune.py 画图
        # 现在它们代表正式的五折 Test 平均结果
        "Avg_RMSE": avg_test_rmse,
        "Avg_MAE": avg_test_mae,
        "Avg_R2": avg_test_r2,
        "Avg_PCC": avg_test_pcc,
        "Avg_SCC": avg_test_scc,

        # 对应五折 Test 标准差
        "Std_RMSE": std_test_rmse,
        "Std_MAE": std_test_mae,
        "Std_R2": std_test_r2,
        "Std_PCC": std_test_pcc,
        "Std_SCC": std_test_scc,

        # 明确写出 Test 字段，方便区分
        "Avg_Test_RMSE": avg_test_rmse,
        "Avg_Test_MAE": avg_test_mae,
        "Avg_Test_R2": avg_test_r2,
        "Avg_Test_PCC": avg_test_pcc,
        "Avg_Test_SCC": avg_test_scc,
        "Std_Test_RMSE": std_test_rmse,
        "Std_Test_MAE": std_test_mae,
        "Std_Test_R2": std_test_r2,
        "Std_Test_PCC": std_test_pcc,
        "Std_Test_SCC": std_test_scc,

        # 保留 Val 平均结果，方便检查，但论文正式结果建议使用 Test
        "Avg_Best_Val_RMSE": avg_val_rmse,
        "Avg_Best_Val_MAE": avg_val_mae,
        "Avg_Best_Val_R2": avg_val_r2,
        "Avg_Best_Val_PCC": avg_val_pcc,
        "Avg_Best_Val_SCC": avg_val_scc,
        "Std_Best_Val_RMSE": std_val_rmse,
        "Std_Best_Val_MAE": std_val_mae,
        "Std_Best_Val_R2": std_val_r2,
        "Std_Best_Val_PCC": std_val_pcc,
        "Std_Best_Val_SCC": std_val_scc
    }

    res_dir = os.path.join('result', dataset_tag, exp_dim)
    os.makedirs(res_dir, exist_ok=True)

    # 保存每折详细结果
    pd.DataFrame(all_fold_records).to_csv(
        os.path.join(res_dir, f"{base_name}_fold_details.csv"),
        index=False,
        encoding='utf-8-sig'
    )

    # 保存五折平均汇总结果
    pd.DataFrame([avg_results]).to_csv(
        os.path.join(res_dir, f"{base_name}_summary.csv"),
        index=False,
        encoding='utf-8-sig'
    )

    print("\n========== 五折独立 Test 平均结果 ==========")
    print(
        f"Test RMSE: {avg_results['Avg_Test_RMSE']:.4f} ± {avg_results['Std_Test_RMSE']:.4f} | "
        f"Test MAE: {avg_results['Avg_Test_MAE']:.4f} ± {avg_results['Std_Test_MAE']:.4f} | "
        f"Test R2: {avg_results['Avg_Test_R2']:.4f} ± {avg_results['Std_Test_R2']:.4f} | "
        f"Test PCC: {avg_results['Avg_Test_PCC']:.4f} ± {avg_results['Std_Test_PCC']:.4f} | "
        f"Test SCC: {avg_results['Avg_Test_SCC']:.4f} ± {avg_results['Std_Test_SCC']:.4f}"
    )

    print("\n========== 五折 Best Val 平均结果，仅供参考 ==========")
    print(
        f"Best Val RMSE: {avg_results['Avg_Best_Val_RMSE']:.4f} ± {avg_results['Std_Best_Val_RMSE']:.4f} | "
        f"Best Val MAE: {avg_results['Avg_Best_Val_MAE']:.4f} ± {avg_results['Std_Best_Val_MAE']:.4f} | "
        f"Best Val R2: {avg_results['Avg_Best_Val_R2']:.4f} ± {avg_results['Std_Best_Val_R2']:.4f} | "
        f"Best Val PCC: {avg_results['Avg_Best_Val_PCC']:.4f} ± {avg_results['Std_Best_Val_PCC']:.4f} | "
        f"Best Val SCC: {avg_results['Avg_Best_Val_SCC']:.4f} ± {avg_results['Std_Best_Val_SCC']:.4f}"
    )

    return avg_results


if __name__ == "__main__":
    args = get_args()
    TraValTes(args)
