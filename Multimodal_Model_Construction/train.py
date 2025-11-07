"""
train.py (已更新适配 45k 数据集)

主训练脚本。
执行 K-Fold 交叉验证来训练和评估多模态融合模型。
保存每个 fold 的最佳模型（基于 val_auc）和相应的注意力权重到 *_45k 目录。
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from tqdm import tqdm
import datetime
import joblib # 用于保存预处理器
from pathlib import Path # 使用 pathlib

# 导入我们的自定义模块
import config # 应该导入更新后的 config.py
import dataset
import model as model_architecture

def train_epoch(model, loader, criterion, optimizer):
    """为单个 epoch 运行训练循环 (不变)"""
    model.train()
    running_loss = 0.0
    for (micro, macro, text), labels in loader:
        micro, macro, text = micro.to(config.DEVICE), macro.to(config.DEVICE), text.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        optimizer.zero_grad()
        logits, _ = model(micro, macro, text)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * micro.size(0)
    return running_loss / len(loader.dataset)

def validate_epoch(model, loader, criterion):
    """为单个 epoch 运行验证循环 (不变)"""
    model.eval()
    all_val_logits, all_val_labels, all_val_attn_weights = [], [], []
    with torch.no_grad():
        for (micro, macro, text), labels in loader:
            micro, macro, text = micro.to(config.DEVICE), macro.to(config.DEVICE), text.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            logits, att_weights = model(micro, macro, text)
            all_val_logits.append(logits)
            all_val_labels.append(labels)
            all_val_attn_weights.append(att_weights)
            
    all_val_logits = torch.cat(all_val_logits).cpu()
    all_val_labels = torch.cat(all_val_labels).cpu()
    all_val_attn_weights = torch.cat(all_val_attn_weights).cpu().numpy()
    
    probs = torch.sigmoid(all_val_logits).numpy()
    preds = (probs > 0.5).astype(int)
    labels_np = all_val_labels.numpy()
    
    try:
        val_auc = roc_auc_score(labels_np, probs)
    except ValueError:
        val_auc = 0.5 # 处理只有一类的情况
    
    val_loss = criterion(all_val_logits, all_val_labels).item()
    val_acc = accuracy_score(labels_np, preds)
    val_mcc = matthews_corrcoef(labels_np, preds)
    
    return val_loss, val_auc, val_acc, val_mcc, all_val_attn_weights, labels_np

def run_training():
    """主训练函数"""
    print(f"--- 实验开始 (45k 数据集) ---")
    print(f"设备: {config.DEVICE}")
    print(f"数据目录: {config.DATA_DIR}")
    print(f"K-Folds: {config.N_SPLITS}, Epochs: {config.EPOCHS}, Batch Size: {config.BATCH_SIZE}")

    # 确保输出目录存在 (使用 config.py 中更新的路径)
    config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    config.ATTENTION_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # 总结果目录

    # 1. 加载原始数据 (使用 config.py 更新的路径)
    print("加载 45k 数据集...")
    X_micro, X_macro, X_text, y = dataset.load_data()

    # 2. 初始化 K-Fold
    kf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=42)

    # 存储每个 fold 的最佳指标
    fold_best_aucs, fold_best_accs, fold_best_mccs = [], [], []

    # 3. K-Fold 循环
    for fold, (train_indices, val_indices) in enumerate(kf.split(X_micro, y)):
        fold_num = fold + 1
        print(f"\n--- Starting Fold {fold_num}/{config.N_SPLITS} ---")

        # A. 预处理
        preprocessor_micro, preprocessor_macro, preprocessor_text = dataset.get_preprocessors()
        
        print(f"  > Fitting preprocessors for Fold {fold_num}...")
        preprocessor_micro.fit(X_micro.iloc[train_indices])
        preprocessor_macro.fit(X_macro.iloc[train_indices])
        preprocessor_text.fit(X_text.iloc[train_indices])
        
        # *** 修改: 保存预处理器到 _45k 目录 ***
        preprocessor_save_path = config.MODEL_SAVE_DIR / f"preprocessor_micro_45k_fold_{fold_num}.pkl"
        joblib.dump(preprocessor_micro, preprocessor_save_path)
        print(f"  > Micro preprocessor for Fold {fold_num} saved to {preprocessor_save_path}")

        X_micro_train = preprocessor_micro.transform(X_micro.iloc[train_indices])
        X_micro_val = preprocessor_micro.transform(X_micro.iloc[val_indices])
        X_macro_train = preprocessor_macro.transform(X_macro.iloc[train_indices])
        X_macro_val = preprocessor_macro.transform(X_macro.iloc[val_indices])
        X_text_train = preprocessor_text.transform(X_text.iloc[train_indices])
        X_text_val = preprocessor_text.transform(X_text.iloc[val_indices])
        y_train, y_val = y.iloc[train_indices].values, y.iloc[val_indices].values
        
        # 动态维度 (逻辑不变)
        current_micro_dim = X_micro_train.shape[1]
        print(f"  > Fold {fold_num} Micro OHE Dim: {current_micro_dim}")

        # B. DataLoaders (使用 config.py 更新的 BATCH_SIZE)
        train_loader, val_loader = dataset.get_data_loaders(
            X_micro_train, X_macro_train, X_text_train, y_train,
            X_micro_val, X_macro_val, X_text_val, y_val,
            batch_size=config.BATCH_SIZE
        )

        # C. 模型初始化 (使用动态维度和 config.py 更新的 DROPOUT_RATE)
        model = model_architecture.MultiModalFusionModel(
            micro_dim=current_micro_dim, 
            macro_dim=X_macro.shape[1],
            text_dim=X_text.shape[1],
            embed_dim=config.EMBED_DIM,
            n_heads=4, 
            dropout=config.DROPOUT_RATE
        ).to(config.DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss() 

        # D. Epoch 循环跟踪变量
        best_val_auc = -1.0
        best_epoch_metrics = (0, 0, 0) # acc, mcc, auc
        best_attention_weights, best_labels = None, None
        
        # *** 修改: 更新保存文件名 ***
        model_save_path = config.MODEL_SAVE_DIR / f"best_model_45k_fold_{fold_num}.pth"
        attn_save_path = config.ATTENTION_DIR / f"best_attention_45k_fold_{fold_num}.npz"

        # D. Epoch 循环
        epoch_iterator = tqdm(range(config.EPOCHS), desc=f"Fold {fold_num} Epochs", leave=False)
        for epoch in epoch_iterator:
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_auc, val_acc, val_mcc, att_weights, val_labels = validate_epoch(model, val_loader, criterion)
            
            epoch_iterator.set_description(
                f"Fold {fold_num} | E {epoch+1:02d} | Tr L: {train_loss:.3f} | V AUC: {val_auc:.3f} | V ACC: {val_acc:.3f}"
            )

            # E. 保存最佳模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch_metrics = (val_acc, val_mcc, val_auc)
                best_attention_weights = att_weights
                best_labels = val_labels
                torch.save(model.state_dict(), model_save_path)

        # F. 保存最佳 epoch 的注意力和指标
        print(f"\nFold {fold_num} complete. Best Val AUC: {best_val_auc:.4f}")
        # 确保 best_attention_weights 不是 None (如果第一个 epoch 就是最好的)
        if best_attention_weights is not None:
             np.savez(attn_save_path, weights=best_attention_weights, labels=best_labels)
             print(f"  > Saving best attention weights to {attn_save_path}")
        else:
             print(f"  > 警告:未能保存注意力权重 (best_attention_weights is None)。")
        print(f"  > Saving best model to {model_save_path}")
        
        fold_best_accs.append(best_epoch_metrics[0])
        fold_best_mccs.append(best_epoch_metrics[1])
        fold_best_aucs.append(best_epoch_metrics[2])

    # 4. 总结所有 Folds
    print("\n--- Cross-Validation Summary (45k Data) ---")
    
    acc_mean, acc_std = np.mean(fold_best_accs), np.std(fold_best_accs)
    mcc_mean, mcc_std = np.mean(fold_best_mccs), np.std(fold_best_mccs)
    auc_mean, auc_std = np.mean(fold_best_aucs), np.std(fold_best_aucs)

    print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"MCC:      {mcc_mean:.4f} ± {mcc_std:.4f}")
    print(f"ROC AUC:  {auc_mean:.4f} ± {auc_std:.4f}")

    # 5. 保存总结报告
    summary_df = pd.DataFrame({
        "metric": ["Accuracy", "MCC", "ROC_AUC"],
        "mean": [acc_mean, mcc_mean, auc_mean],
        "std_dev": [acc_std, mcc_std, auc_std],
        "result_str": [f"{acc_mean:.4f} ± {acc_std:.4f}", f"{mcc_mean:.4f} ± {mcc_std:.4f}", f"{auc_mean:.4f} ± {auc_std:.4f}"]
    })

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # *** 修改: 更新总结文件名 ***
    summary_filename = f"nn_fusion_model_summary_45k_{timestamp}.csv"
    summary_path = config.OUTPUT_DIR / summary_filename # 使用总结果目录
    
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary report saved to: {summary_path}")

if __name__ == "__main__":
    try:
        run_training()
    except Exception as e:
        print(f"\n--- An error occurred ---")
        import traceback
        traceback.print_exc()