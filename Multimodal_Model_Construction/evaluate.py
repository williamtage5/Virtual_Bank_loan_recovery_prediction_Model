"""
evaluate.py (已修复 tqdm 导入)

加载一个训练好的 K-Fold 模型 *及其对应的预处理器* (来自 45k 训练)，
在 *整个* 45k 数据集上运行它，并保存最终的注意力权重、标签和 logits。

同时，通过搜索最佳阈值来优化决策指标 (ACC, MCC)。
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
# import matplotlib.pyplot as plt # 如果需要绘图，取消注释
import joblib 
from pathlib import Path 
# *** 修复：修改 tqdm 导入方式 ***
from tqdm import tqdm 

# 导入我们的自定义模块
import config # 应该导入更新后的 config.py (指向 45k 路径)
import dataset
import model as model_architecture 

def evaluate_model(model_path, preprocessors, data):
    """加载模型并在所有数据上运行评估。"""
    
    X_micro, X_macro, X_text, y = data
    
    print("  > Transforming data using loaded/fitted preprocessors...")
    try:
        X_micro_processed = preprocessors['micro'].transform(X_micro)
        X_macro_processed = preprocessors['macro'].transform(X_macro)
        X_text_processed = preprocessors['text'].transform(X_text)
        y_values = y.values
    except Exception as e:
         print(f"  > 错误：转换数据时失败: {e}"); raise
         
    current_micro_dim = X_micro_processed.shape[1]
    print(f"  > Loaded preprocessor produced micro dim: {current_micro_dim}")
    
    print(f"  > Loading model state from {model_path}...")
    try:
        eval_model = model_architecture.MultiModalFusionModel(
            micro_dim=current_micro_dim, 
            macro_dim=X_macro.shape[1],
            text_dim=X_text.shape[1],
            embed_dim=config.EMBED_DIM,
            n_heads=4, 
            dropout=0.0 
        ).to(config.DEVICE)
        
        eval_model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        eval_model.eval()
    except Exception as e:
        print(f"  > 错误：加载或初始化模型时失败: {e}"); raise

    print("  > Creating DataLoader for full dataset...")
    full_dataset = dataset.LoanMultiModalDataset(
        X_micro_processed, X_macro_processed, X_text_processed, y_values
    )
    loader = torch.utils.data.DataLoader(
        full_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )

    print("  > Running model inference on full dataset...")
    all_logits, all_labels, all_att_weights = [], [], []
    
    # *** 修复：现在可以直接使用 tqdm() ***
    inference_progress = tqdm(loader, desc="Inference") 

    with torch.no_grad():
        for (micro, macro, text), labels in inference_progress:
            micro, macro, text = micro.to(config.DEVICE), macro.to(config.DEVICE), text.to(config.DEVICE)
            
            logits, att_weights = eval_model(micro, macro, text)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu()) 
            all_att_weights.append(att_weights.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_att_weights = torch.cat(all_att_weights).numpy()
    
    print("  > Inference complete.")
    return all_logits, all_labels, all_att_weights

def find_optimal_threshold(logits, labels):
    """搜索最佳决策阈值以最大化 MCC。(逻辑不变)"""
    print("\n--- Finding Optimal Threshold ---")
    
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    
    try:
        default_auc = roc_auc_score(labels_np, probs)
    except ValueError: default_auc = 0.5 
        
    default_preds = (probs > 0.5).astype(int)
    default_acc = accuracy_score(labels_np, default_preds)
    default_mcc = matthews_corrcoef(labels_np, default_preds)
    
    print(f"Metrics @ 0.5 Threshold: AUC={default_auc:.4f}, ACC={default_acc:.4f}, MCC={default_mcc:.4f}")

    print("  > Searching for best threshold...")
    thresholds = np.linspace(0.01, 0.99, 100)
    # 添加 try-except 以处理 MCC 可能计算失败的情况 (例如全零预测)
    mcc_scores = []
    for t in thresholds:
        try:
             preds_t = (probs > t).astype(int)
             mcc_t = matthews_corrcoef(labels_np, preds_t)
             mcc_scores.append(mcc_t)
        except Exception: # 捕获任何可能的错误
             mcc_scores.append(-1) # 给定一个无效分数
             
    best_idx = np.argmax(mcc_scores)
    best_threshold = thresholds[best_idx]
    best_mcc = mcc_scores[best_idx]
    
    best_preds = (probs > best_threshold).astype(int)
    best_acc = accuracy_score(labels_np, best_preds)
    
    print(f"\nBest Threshold (found by maximizing MCC): {best_threshold:.3f}") 
    print(f"Metrics @ Best Threshold ({best_threshold:.3f}):  AUC={default_auc:.4f}, ACC={best_acc:.4f}, MCC={best_mcc:.4f}")
    
    return best_threshold

def main():
    print("--- 运行最终评估和注意力提取脚本 (45k 版本) ---")
    
    print("Loading 45k dataset...")
    try:
        X_micro, X_macro, X_text, y = dataset.load_data()
        print(f"  > Dataset loaded successfully. Shape X_micro: {X_micro.shape}")
    except Exception as e:
        print(f"  > 错误：加载数据失败: {e}"); return

    fold_to_evaluate = 1 
    preprocessor_filename = f"preprocessor_micro_45k_fold_{fold_to_evaluate}.pkl"
    preprocessor_micro_path = config.MODEL_SAVE_DIR / preprocessor_filename 
    
    if not preprocessor_micro_path.exists():
        print(f"[ERROR] 预处理器文件未找到: {preprocessor_micro_path}"); return
        
    print(f"Loading preprocessor from {preprocessor_micro_path}...")
    try:
        preprocessor_micro = joblib.load(preprocessor_micro_path)
    except Exception as e:
        print(f"  > 错误：加载预处理器失败: {e}"); return
        
    print("Fitting macro and text preprocessors on full 45k data...")
    try:
        _, preprocessor_macro, preprocessor_text = dataset.get_preprocessors()
        preprocessor_macro.fit(X_macro)
        preprocessor_text.fit(X_text)
    except Exception as e:
        print(f"  > 错误：拟合 macro/text 预处理器失败: {e}"); return
        
    preprocessors = {'micro': preprocessor_micro, 'macro': preprocessor_macro, 'text': preprocessor_text}
    
    model_filename = f"best_model_45k_fold_{fold_to_evaluate}.pth"
    model_to_load = config.MODEL_SAVE_DIR / model_filename 
    if not model_to_load.exists():
        print(f"[ERROR] 模型文件未找到: {model_to_load}"); return
        
    try:
        all_logits, all_labels, all_att_weights = evaluate_model(
            model_to_load, preprocessors, (X_micro, X_macro, X_text, y))
    except Exception as e:
        print(f"[ERROR] 模型评估过程中发生错误: {e}")
        import traceback; traceback.print_exc(); return 

    print("\n--- Saving Final Outputs ---")
    output_npz_filename = f"final_attention_45k_fold_{fold_to_evaluate}.npz"
    save_path = config.ATTENTION_DIR / output_npz_filename 
    
    try:
        np.savez(save_path, weights=all_att_weights, labels=all_labels.numpy(), logits=all_logits.numpy())
        print(f"Successfully saved {len(all_labels)} samples to: {save_path}")
    except Exception as e:
        print(f"  > 错误：保存 NPZ 文件时失败: {e}")

    find_optimal_threshold(all_logits, all_labels)

if __name__ == "__main__":
    main()