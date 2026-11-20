import torch
import argparse
import pandas as pd
import os
from tqdm import tqdm
import fm

# 修复 PyTorch 2.6+ 加载 RNA-FM 模型权重的安全限制
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([argparse.Namespace])

# 全局设备判断
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ✅ 默认参数已锁定为：窗口=3，步长=1（重叠滑动）
def extract_features(raw_dir, out_path_orig, out_path_window, model_path, window=3, stride=1):
    # 1. 动态拼接输入文件路径
    rna_file = os.path.join(raw_dir, 'RNA.xlsx')

    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    model, alphabet = fm.pretrained.rna_fm_t12(model_location=model_path)

    model = model.to(DEVICE)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    # 3. 读取数据
    print(f">>> 正在读取数据：{rna_file}")
    df = pd.read_excel(rna_file)

    orig_embeddings = {}
    window_embeddings = {}

    # RNA-FM 的最大长度限制 (1024 - 2个特殊标记)
    max_len = 1022

    print(f"🧬 开始提取 RNA Embedding (Window={window}, Stride={stride})...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # 统一转为 int，保持字典键格式一致
                rna_id = int(row['RNA_ID'])

                # 读取序列并清洗（统一大写、去首尾空格、T转U、只保留ACGU）
                raw_seq = str(row['RNA_Sequence']).upper().strip().replace('T', 'U')
                seq = "".join([nt for nt in raw_seq if nt in 'ACGU'])

                # 物理截断与窗口大小判断
                if not seq:
                    continue
                if len(seq) > max_len:
                    seq = seq[:max_len]

                # 如果清洗后的序列比窗口（3个碱基）还短，则跳过
                if len(seq) < window:
                    print(f"⚠️ 跳过 RNA_ID {rna_id}: 序列长度({len(seq)})小于窗口大小({window})")
                    continue

                # 转换为模型输入张量
                data = [(str(rna_id), seq)]
                _, _, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(DEVICE)

                # 前向传播提取第 12 层特征
                results = model(batch_tokens, repr_layers=[12])
                token_embeddings = results["representations"][12]

                # 提取纯碱基特征：移除第0位的 <cls> 和最后一位的 <eos>
                L = len(seq)
                base_features = token_embeddings[0, 1:L + 1, :]  # 形状: [L, 640]

                # 滑动窗口切片与池化 (核心逻辑：unfold提取窗口，mean计算平均值)
                windows = base_features.unfold(0, window, stride)  # 形状: [num_windows, 640, 3]
                augmented_matrix = windows.mean(dim=2)  # 形状: [num_windows, 640]

                # 转移到 CPU 并转为半精度以节省磁盘空间
                orig_embeddings[rna_id] = base_features.cpu().half()
                window_embeddings[rna_id] = augmented_matrix.cpu().half()

            except Exception as e:
                raise RuntimeError(f"RNA {rna_id} extraction failed") from e

    # 4. 确保输出目录存在并保存文件
    os.makedirs(os.path.dirname(out_path_orig), exist_ok=True)
    os.makedirs(os.path.dirname(out_path_window), exist_ok=True)

    torch.save(orig_embeddings, out_path_orig)
    print(f"✅ 原始表示已保存: {out_path_orig}")

    torch.save(window_embeddings, out_path_window)
    print(f"✅ 窗口平均表示已保存: {out_path_window}")


# 单独运行测试
if __name__ == "__main__":
    extract_features(
        raw_dir=r'data/raw',
        out_path_orig=r'data/processed/extracted/rna_orig.pt',
        out_path_window=r'data/processed/extracted/rna_window.pt',
        model_path=r"models/RNA-FM_pretrained.pth",
        window=3,  # 窗口大小：3
        stride=1  # 步长：1（重叠滑动）
    )