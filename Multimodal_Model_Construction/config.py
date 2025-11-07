"""
config.py (已更新适配 45k 数据集)

集中配置文件，存储所有路径、超参数和特征列表，供项目中的所有其他脚本导入。
"""

import torch
from pathlib import Path

# --- 1. 核心路径 ---
BASE_DIR = Path(__file__).resolve().parent.parent
# *** 修改: 指向新的 45k 数据目录 ***
DATA_DIR = BASE_DIR / "data" / "3_parts_segment_45K" 
OUTPUT_DIR = BASE_DIR / "data" / "experiment_result"

# --- 2. 数据文件路径 ---
# *** 修改: 更新文件名以匹配 45k 数据 ***
MICRO_DATA_PATH = DATA_DIR / "micro_data_45k.csv"
MACRO_DATA_PATH = DATA_DIR / "macro_data_45k.csv"
TEXT_EMBED_PATH = DATA_DIR / "text_embeddings_45k.csv"

# --- 3. 输出路径 ---
# *** 修改: 添加 _45k 后缀以区分 ***
MODEL_SAVE_DIR = OUTPUT_DIR / "models_45k" 
ATTENTION_DIR = OUTPUT_DIR / "attention_plots_45k"

# --- 4. 微观特征列表 (保持不变) ---
# (这些列表定义了 *哪些* 列属于数值/类别，与数据量无关)

# 4a. 宏观特征 (用于从数值列表中排除它们)
MACRO_FEATURES = [
    "All_Employees(Total_Nonfarm)", "Building_Permits", "Consumer_Confidence",
    "Consumer_Price Index", "Consumer_Sentiment", "Federal_Funds_Effective_Rate",
    "Industrial_Production", "Labor_Force_Participation_Rate",
    "Manufacturers_New_Orders: Durable Goods", "Money_Supply_(M2)",
    "National_Home_Price_Index", "Personal_Consumption_Expenditures",
    "Personal_Income", "Producer_Price_Index", "Retail_Sales",
    "Trade_Balance", "Unemployment_Rate"
]

# 4b. 包含宏观特征的完整数值列表
ALL_NUMERICAL_FEATURES = [
    "All_Employees(Total_Nonfarm)", "Building_Permits", "Consumer_Confidence",
    "Consumer_Price Index", "Consumer_Sentiment", "Federal_Funds_Effective_Rate",
    "Industrial_Production", "Labor_Force_Participation_Rate",
    "Manufacturers_New_Orders: Durable Goods", "Money_Supply_(M2)",
    "National_Home_Price_Index", "Personal_Consumption_Expenditures",
    "Personal_Income", "Producer_Price_Index", "Retail_Sales",
    "Trade_Balance", "Unemployment_Rate", "all_util", "annual_inc",
    "avg_cur_bal", "bc_open_to_buy", "bc_util",
    "credit_history_months_engineered", "dti", "il_util", "installment",
    "int_rate", "loan_amnt", "loan_to_income", "max_bal_bc",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl", "mths_since_rcnt_il", "mths_since_recent_bc",
    "pct_tl_nvr_dlq", "percent_bc_gt_75", "revol_bal", "revol_util",
    "tot_coll_amt", "tot_cur_bal", "tot_hi_cred_lim", "total_bal_ex_mort",
    "total_bal_il", "total_bc_limit", "total_debt",
    "total_il_high_credit_limit", "total_rev_hi_lim"
]

# 4c. 最终的微观特征列表 (逻辑不变)
MICRO_NUMERICAL_FEATURES = [f for f in ALL_NUMERICAL_FEATURES if f not in MACRO_FEATURES]
MICRO_CATEGORICAL_FEATURES = [
    "acc_now_delinq", "acc_open_past_24mths", "addr_state", "application_type",
    "chargeoff_within_12_mths", "collections_12_mths_ex_med", "delinq_2yrs",
    "delinq_amnt", "disbursement_method", "emp_length", "fico_range_high",
    "fico_range_low", "grade", "home_ownership", "initial_list_status",
    "inq_fi", "inq_last_12m", "inq_last_6mths", "mort_acc",
    "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
    "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "open_acc", "open_acc_6m", "open_act_il",
    "open_il_12m", "open_il_24m", "open_rv_12m", "open_rv_24m", "pub_rec",
    "pub_rec_bankruptcies", "purpose", "sub_grade", "tax_liens", "term",
    "total_acc", "total_cu_tl", "verification_status"
]


# --- 5. 模型超参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# *** 修改: 增加 BATCH_SIZE ***
BATCH_SIZE = 128  # 增加批量大小以适应更大的数据集
EPOCHS = 50       # 保持 Epoch 不变，可以稍后调整
LEARNING_RATE = 1e-4 # 保持学习率不变
N_SPLITS = 5      # K-Fold 折数保持不变
# *** 修改: 降低 DROPOUT_RATE ***
DROPOUT_RATE = 0.2  # 数据量更大，可以稍微降低 Dropout
EMBED_DIM = 32    # 三个塔的统一输出维度保持不变

# *** 修改: 删除旧的、硬编码的维度 ***
# MICRO_OHE_FEATURES_DIM = 453 # <--- 删除或注释掉这一行


# --- 6. 调试信息 (可选) ---
if __name__ == "__main__":
    print(f"--- Configuration File (45k Version) ---")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"DEVICE: {DEVICE}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"DROPOUT_RATE: {DROPOUT_RATE}")
    print(f"Micro numerical features count: {len(MICRO_NUMERICAL_FEATURES)}")
    print(f"Micro categorical features count: {len(MICRO_CATEGORICAL_FEATURES)}")
    # 检查路径是否存在
    print(f"Micro data path exists: {MICRO_DATA_PATH.exists()}")
    print(f"Macro data path exists: {MACRO_DATA_PATH.exists()}")
    print(f"Text embed path exists: {TEXT_EMBED_PATH.exists()}")
    print(f"Config file loaded successfully.")