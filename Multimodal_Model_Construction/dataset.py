"""
dataset.py (已再次更新 - 添加 Inf 替换)

负责加载、预处理和打包数据，以便输入到多模态PyTorch模型中。
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
import config # 导入更新后的 config.py

def load_data():
    """
    加载三个 *_45k.csv 文件，合并，并清理 Inf 值。
    """
    print("--- 开始加载 45k 数据集 ---")
    # ... (加载 df_micro, df_macro, df_text 的代码保持不变) ...
    loaded_files = []
    try:
        print(f"  > 加载 Micro: {config.MICRO_DATA_PATH}")
        df_micro = pd.read_csv(config.MICRO_DATA_PATH, low_memory=False)
        loaded_files.append(config.MICRO_DATA_PATH.name)
        print(f"  > 加载 Macro: {config.MACRO_DATA_PATH}")
        df_macro = pd.read_csv(config.MACRO_DATA_PATH, low_memory=False)
        loaded_files.append(config.MACRO_DATA_PATH.name)
        print(f"  > 加载 Text Embeddings: {config.TEXT_EMBED_PATH}")
        df_text = pd.read_csv(config.TEXT_EMBED_PATH, low_memory=False)
        loaded_files.append(config.TEXT_EMBED_PATH.name)
        print(f"--- 数据文件加载成功: {loaded_files} ---")
    except FileNotFoundError as e:
        print(f"[Error] 数据文件未找到: {e}"); raise
    except Exception as e:
         print(f"[Error] 加载数据文件时发生未知错误: {e}"); raise

    # ... (合并 df_merged 的代码保持不变) ...
    print("--- 开始合并数据 ---")
    try:
        df_merged = pd.merge(df_micro, df_macro, on=['unique_loan_id', 'target'], how='inner')
        initial_merge_count = len(df_merged)
        print(f"  > Micro 与 Macro 合并后: {initial_merge_count} 条记录")
        df_merged = pd.merge(df_merged, df_text, on=['unique_loan_id', 'target'], how='inner')
        final_merge_count = len(df_merged)
        print(f"  > ...与 Text 合并后: {final_merge_count} 条记录")
        if final_merge_count < initial_merge_count or final_merge_count < len(df_micro):
            print("警告：合并过程中丢失了部分记录，请检查 unique_loan_id 和 target 是否在所有文件中完全匹配。")
        print("--- 数据合并成功 ---")
    except KeyError as ke:
         print(f"[Error] 合并失败，缺少关键列: {ke}"); raise
    except Exception as e:
         print(f"[Error] 合并数据时发生未知错误: {e}"); raise

    # ... (准备 X_micro, X_macro, X_text, y 的代码保持不变) ...
    print("--- 准备 X 和 y 数据 ---")
    try:
        micro_num_cols_in_df = [col for col in config.MICRO_NUMERICAL_FEATURES if col in df_merged.columns]
        micro_cat_cols_in_df = [col for col in config.MICRO_CATEGORICAL_FEATURES if col in df_merged.columns]
        X_micro = df_merged[micro_num_cols_in_df + micro_cat_cols_in_df].copy() # 使用 .copy() 避免 SettingWithCopyWarning
        
        macro_cols_in_df = [col for col in config.MACRO_FEATURES if col in df_merged.columns]
        X_macro = df_merged[macro_cols_in_df].copy()
        
        text_cols_in_df = [col for col in df_text.columns if col not in ['unique_loan_id', 'target']]
        X_text = df_merged[text_cols_in_df].copy()
        
        y = df_merged['target']
        
        print(f"  > X_micro 形状: {X_micro.shape}")
        print(f"  > X_macro 形状: {X_macro.shape}")
        print(f"  > X_text 形状: {X_text.shape}")
        print(f"  > y 形状: {y.shape}")

    except KeyError as ke:
        print(f"[Error] 准备 X, y 时选取列失败，缺少列: {ke}"); raise
    except Exception as e:
        print(f"[Error] 准备 X, y 时发生未知错误: {e}"); raise

    # *** 新增：在这里检查并替换所有数值列中的 Inf 值 ***
    print("--- 检查并替换 Inf 值 ---")
    inf_found_count = 0
    # 检查 Micro 数值列
    for col in micro_num_cols_in_df:
        if np.isinf(X_micro[col]).any():
            count = np.isinf(X_micro[col]).sum()
            print(f"  > 在 X_micro 列 '{col}' 中发现 {count} 个 Inf 值，替换为 NaN。")
            X_micro[col] = X_micro[col].replace([np.inf, -np.inf], np.nan)
            inf_found_count += count
    # 检查 Macro 列
    for col in macro_cols_in_df:
         if np.isinf(X_macro[col]).any():
            count = np.isinf(X_macro[col]).sum()
            print(f"  > 在 X_macro 列 '{col}' 中发现 {count} 个 Inf 值，替换为 NaN。")
            X_macro[col] = X_macro[col].replace([np.inf, -np.inf], np.nan)
            inf_found_count += count
    # 检查 Text Embedding 列
    for col in text_cols_in_df:
         if np.isinf(X_text[col]).any():
            count = np.isinf(X_text[col]).sum()
            print(f"  > 在 X_text 列 '{col}' 中发现 {count} 个 Inf 值，替换为 NaN。")
            X_text[col] = X_text[col].replace([np.inf, -np.inf], np.nan)
            inf_found_count += count
            
    if inf_found_count == 0:
        print("  > 未在任何数值列中发现 Inf 值。")
    # ******************************************************
        
    return X_micro, X_macro, X_text, y

# --- get_preprocessors(), LoanMultiModalDataset(), get_data_loaders() ---
# (这些函数的代码保持不变)
def get_preprocessors():
    numeric_pipe_linear = Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
    categorical_pipe_linear = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor_micro = ColumnTransformer(
        transformers=[('num', numeric_pipe_linear, config.MICRO_NUMERICAL_FEATURES),('cat', categorical_pipe_linear, config.MICRO_CATEGORICAL_FEATURES)],
        remainder='passthrough', n_jobs=-1)
    preprocessor_macro = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
    preprocessor_text = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    return preprocessor_micro, preprocessor_macro, preprocessor_text

class LoanMultiModalDataset(Dataset):
    def __init__(self, micro_data, macro_data, text_data, labels):
        self.micro_data = torch.tensor(micro_data, dtype=torch.float32)
        self.macro_data = torch.tensor(macro_data, dtype=torch.float32)
        self.text_data = torch.tensor(text_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        inputs = (self.micro_data[idx], self.macro_data[idx], self.text_data[idx])
        label = self.labels[idx]
        return inputs, label

def get_data_loaders(X_micro_train, X_macro_train, X_text_train, y_train, X_micro_val, X_macro_val, X_text_val, y_val, batch_size):
    train_dataset = LoanMultiModalDataset(X_micro_train, X_macro_train, X_text_train, y_train)
    val_dataset = LoanMultiModalDataset(X_micro_val, X_macro_val, X_text_val, y_val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) 
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) 
    return train_loader, val_loader


# --- 调试脚本 ---
# (保持不变，除了移除写入 config.py 的部分)
if __name__ == "__main__":
    print("[DEBUG] 正在运行 dataset.py 调试脚本 (45k 版本, Inf Fix)...")
    try:
        X_micro, X_macro, X_text, y = load_data()
        print(f"[DEBUG] 原始数据已加载。总样本数: {len(y)}")
    except Exception as e:
        print(f"[FATAL] 加载数据失败: {e}"); exit() 
    kf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=42)
    try:
        train_indices, val_indices = next(kf.split(X_micro, y))
    except Exception as e:
        print(f"[FATAL] KFold 拆分失败: {e}"); exit()
    print(f"[DEBUG] 模拟 K-Fold 拆分: {len(train_indices)} 训练, {len(val_indices)} 验证。")
    try:
        preprocessor_micro, preprocessor_macro, preprocessor_text = get_preprocessors()
        print("[DEBUG] 正在拟合 (fit) 预处理器...")
        X_micro_train_df = X_micro.iloc[train_indices]
        preprocessor_micro.fit(X_micro_train_df)
        X_macro_train_df = X_macro.iloc[train_indices]
        preprocessor_macro.fit(X_macro_train_df)
        X_text_train_df = X_text.iloc[train_indices]
        preprocessor_text.fit(X_text_train_df)
        print("[DEBUG] 预处理器拟合完毕。")
    except Exception as e:
        print(f"[FATAL] 拟合预处理器失败: {e}")
        import traceback
        traceback.print_exc() # 打印详细错误
        exit()
    print("[DEBUG] 正在转换 (transform) 数据...")
    try:
        X_micro_train = preprocessor_micro.transform(X_micro_train_df)
        X_micro_val = preprocessor_micro.transform(X_micro.iloc[val_indices]) 
        X_macro_train = preprocessor_macro.transform(X_macro_train_df)
        X_macro_val = preprocessor_macro.transform(X_macro.iloc[val_indices]) 
        X_text_train = preprocessor_text.transform(X_text_train_df)
        X_text_val = preprocessor_text.transform(X_text.iloc[val_indices]) 
        y_train = y.iloc[train_indices].values
        y_val = y.iloc[val_indices].values
        print("[DEBUG] 数据转换完毕。")
    except Exception as e:
        print(f"[FATAL] 转换数据失败: {e}"); exit()
    try:
        train_loader, val_loader = get_data_loaders(
            X_micro_train, X_macro_train, X_text_train, y_train,
            X_micro_val, X_macro_val, X_text_val, y_val,
            batch_size=config.BATCH_SIZE
        )
        print("[DEBUG] Data loaders created successfully.")
    except Exception as e:
        print(f"[FATAL] 创建 DataLoader 失败: {e}"); exit()
    print("[DEBUG] Fetching one batch from train_loader...")
    try:
        (micro_batch, macro_batch, text_batch), labels_batch = next(iter(train_loader))
        ohe_features_count = micro_batch.shape[1]
        print(f"[DEBUG] Micro data shape: {micro_batch.shape}") 
        print(f"[DEBUG] Macro data shape: {macro_batch.shape}") 
        print(f"[DEBUG] Text data shape:  {text_batch.shape}")  
        print(f"[DEBUG] Labels shape:     {labels_batch.shape}") 
        print("-" * 30)
        print(f"[INFO] *** 请记录下这个维度用于下一步 ***")
        print(f"[INFO] *** MICRO_OHE_FEATURES_DIM = {ohe_features_count} ***")
        print("-" * 30)
        print("[DEBUG] Debugging complete.")
    except Exception as e:
        print(f"[ERROR] 无法从 DataLoader 获取批次: {e}")
        import traceback
        traceback.print_exc()