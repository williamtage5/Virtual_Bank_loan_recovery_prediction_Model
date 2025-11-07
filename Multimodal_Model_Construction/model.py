"""
model.py (调试脚本已更新)

定义三塔注意力融合神经网络 (MultiModalFusionModel)。
"""

import torch
import torch.nn as nn
import config

class MultiModalFusionModel(nn.Module):
    """
    一个端到端的多模态融合模型。
    """
    def __init__(self, micro_dim, macro_dim, text_dim, embed_dim, n_heads, dropout):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        # --- 1. 特征提取塔 ---
        # Micro-Tower
        self.micro_tower = nn.Sequential(
            nn.Linear(micro_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, embed_dim) )
        # Macro-Tower
        self.macro_tower = nn.Sequential(
            nn.Linear(macro_dim, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(16, embed_dim) )
        # Text-Tower
        self.text_tower = nn.Sequential(
            nn.Linear(text_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, embed_dim) )

        # --- 2. 注意力融合层 ---
        self.attention = nn.MultiheadAttention( # 确认 MultiHeadAttention 拼写正确
            embed_dim=embed_dim, num_heads=n_heads, 
            dropout=dropout, batch_first=True )

        # --- 3. 分类头 ---
        self.classification_head = nn.Sequential(
            nn.Flatten(), nn.Linear(embed_dim * 3, 16), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(16, 1) )

    def forward(self, x_micro, x_macro, x_text):
        """ 前向传播 """
        h_micro = self.micro_tower(x_micro) 
        h_macro = self.macro_tower(x_macro) 
        h_text = self.text_tower(x_text)   

        h_micro_seq = h_micro.unsqueeze(1)
        h_macro_seq = h_macro.unsqueeze(1)
        h_text_seq = h_text.unsqueeze(1)
        x_fused = torch.cat([h_micro_seq, h_macro_seq, h_text_seq], dim=1)

        attn_output, attn_weights = self.attention(
            x_fused, x_fused, x_fused, 
            need_weights=True, average_attn_weights=True )
        
        logits = self.classification_head(attn_output)
        logits = logits.squeeze(-1) # (Batch, 1) -> (Batch)

        return logits, attn_weights

# --- 调试脚本 ---
if __name__ == "__main__":
    print("[DEBUG] 正在运行 model.py 调试脚本 (45k 版本)...")

    # *** 修改：使用 dataset.py 输出的新维度 ***
    micro_dim = 1334 
    print(f"[DEBUG] 使用 Micro OHE Dimension: {micro_dim}")
        
    # 定义其他维度 (保持不变)
    macro_dim = 17
    text_dim = 1024
    
    # 初始化模型 (使用 config.py 中的超参数)
    try:
        model = MultiModalFusionModel(
            micro_dim=micro_dim,
            macro_dim=macro_dim,
            text_dim=text_dim,
            embed_dim=config.EMBED_DIM,
            n_heads=4, # 假设 n_heads=4
            dropout=config.DROPOUT_RATE 
        ).to(config.DEVICE)
        
        model.eval()
        print(f"[DEBUG] 模型已创建并移至 {config.DEVICE}。")
    except Exception as e:
         print(f"[FATAL] 初始化模型失败: {e}")
         exit()

    # 创建虚拟数据 (使用 config.py 中的 BATCH_SIZE)
    batch_size = config.BATCH_SIZE
    print("[DEBUG] 正在创建虚拟数据...")
    try:
        dummy_micro = torch.randn(batch_size, micro_dim).to(config.DEVICE)
        dummy_macro = torch.randn(batch_size, macro_dim).to(config.DEVICE)
        dummy_text = torch.randn(batch_size, text_dim).to(config.DEVICE)
        
        print(f"[DEBUG] Dummy Micro shape: {dummy_micro.shape}") # 应为 (128, 1334)
        print(f"[DEBUG] Dummy Macro shape: {dummy_macro.shape}") # 应为 (128, 17)
        print(f"[DEBUG] Dummy Text shape:  {dummy_text.shape}")  # 应为 (128, 1024)
    except Exception as e:
        print(f"[FATAL] 创建虚拟数据失败: {e}")
        exit()

    # 执行前向传播
    print("[DEBUG] 正在执行前向传播...")
    try:
        with torch.no_grad():
            logits, att_weights = model(dummy_micro, dummy_macro, dummy_text)
        
        print("[DEBUG] 前向传播成功。")
        # 预期 logits: [128] 
        # 预期 att_weights: [128, 3, 3]
        print(f"[DEBUG] Logits output shape: {logits.shape}")
        print(f"[DEBUG] Attention weights output shape: {att_weights.shape}")

        # 验证形状是否正确
        expected_logits_shape = torch.Size([batch_size])
        expected_attn_shape = torch.Size([batch_size, 3, 3])

        if logits.shape == expected_logits_shape and att_weights.shape == expected_attn_shape:
            print("[DEBUG] 调试成功！输出形状符合预期。")
        else:
            print("[ERROR] 输出形状与预期不符！")
            print(f"  > 预期 Logits 形状: {expected_logits_shape}, 得到: {logits.shape}")
            print(f"  > 预期 Attention 形状: {expected_attn_shape}, 得到: {att_weights.shape}")
            
    except Exception as e:
        print(f"[ERROR] 前向传播失败: {e}")
        import traceback
        traceback.print_exc()