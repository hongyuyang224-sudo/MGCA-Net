import torch
import pandas as pd
import os
from tqdm import tqdm
from rinalmo.pretrained import get_pretrained_model

# 设备判断
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_features_with_model(model, alphabet, raw_dir, out_path_orig, out_path_window, window=6, stride=1):
    """
    接收已加载的模型，进行特征提取
    """
    rna_file = os.path.join(raw_dir, 'RNA.xlsx')
    df = pd.read_excel(rna_file)

    # 原始表示只需要提取一次，但为了逻辑简单，我们根据路径是否存在来决定是否保存
    orig_embeddings = {}
    window_embeddings = {}

    print(f"🧬 正在处理 Window={window}, Stride={stride}...")

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"W={window}"):
            rna_id = int(row['RNA_ID'])
            sequence = str(row['RNA_Sequence']).upper().strip()
            seq_len = len(sequence)

            if seq_len < window:
                continue

            tokens = torch.tensor(alphabet.batch_tokenize([sequence]), dtype=torch.int64).to(DEVICE)
            outputs = model(tokens)

            rep = outputs.get("representations", outputs.get("representation"))
            if isinstance(rep, list): rep = rep[-1]
            rep_squeeze = rep.squeeze(0)

            # 这里的逻辑保持不变，处理 CLS/SEP tokens
            if rep_squeeze.shape[0] == seq_len + 2:
                base_features = rep_squeeze[1:-1]
            else:
                base_features = rep_squeeze[:seq_len]

            # --- 核心计算：滑动窗口平均 ---
            # base_features shape: (L, E) -> windows shape: (L-W+1, E, W)
            windows = base_features.unfold(0, window, stride)
            # 对窗口维(dim=2)求平均 -> shape: (L-W+1, E)
            augmented_matrix = windows.mean(dim=2)

            # 只有在路径不存在时才存一份原始的（避免重复占用空间）
            if not os.path.exists(out_path_orig):
                orig_embeddings[rna_id] = rep_squeeze.cpu().half()

            window_embeddings[rna_id] = augmented_matrix.cpu().half()

    # 保存原始表示（仅当文件不存在时）
    if not os.path.exists(out_path_orig) and orig_embeddings:
        os.makedirs(os.path.dirname(out_path_orig), exist_ok=True)
        torch.save(orig_embeddings, out_path_orig)
        print(f"✅ 原始全量表示已保存: {out_path_orig}")

    # 保存窗口表示
    os.makedirs(os.path.dirname(out_path_window), exist_ok=True)
    torch.save(window_embeddings, out_path_window)
    print(f"✅ 窗口平均表示 (W={window}) 已保存: {out_path_window}")


if __name__ == "__main__":
    # 1. 设置路径和参数
    RAW_DATA_DIR = r'data/raw'
    SAVE_DIR = 'data/processed/extracted/rinalmo'

    # 定义你想要测试的窗口大小列表
    window_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    stride = 1

    # 2. 统一加载模型 (只加载一次，节省大量时间)
    print(f"🚀 正在加载 RiNALMo 模型到 {DEVICE}...")
    try:
        model, alphabet = get_pretrained_model(model_name="giga-v1")
        model = model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise

    # 3. 循环生成不同 window 的文件
    orig_path = os.path.join(SAVE_DIR, 'rna_orig_indexed.pt')

    for w in window_sizes:
        # 动态生成文件名，例如: rna_window_mean_w3.pt, rna_window_mean_w6.pt
        win_path = os.path.join(SAVE_DIR, f'rna_window_mean_w{w}.pt')

        extract_features_with_model(
            model=model,
            alphabet=alphabet,
            raw_dir=RAW_DATA_DIR,
            out_path_orig=orig_path,
            out_path_window=win_path,
            window=w,
            stride=stride
        )

    print("\n✨ 所有窗口尺寸的特征提取任务已完成！")