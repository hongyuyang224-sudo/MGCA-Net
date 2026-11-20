import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import random
from tqdm import tqdm
import os
import pandas as pd
import glob
import re
from datetime import datetime
from contextlib import nullcontext

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from models.model import MSGA_RNA


# =========================
# 1. 固定随机种子
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 2. 参数配置
# =========================
def get_args():
    return {
        "seed": 42,

        # 推理 batch size
        "batch_size": 16,

        # 模型结构参数：必须和训练时保持一致
        "lr": 1e-4,
        "weight_decay": 0.1,
        "embed_dim": 128,
        "nhead": 1,
        "fusion_hidden_dim": 64,
        "dropout": 0.1,
        "aux_weight_macro": 0.4,
        "aux_weight_micro": 0.5,
        "scheduler_factor": 0.1,
        "scheduler_patience": 40,
        "early_stop_patience": 60,
        "epochs": 150,
        "n_output": 1,

        # 数据路径
        "data_path": r"data/processed/final_data_random.pt",

        # 原始信息表：如果 .pt 里没有 RNA 名称 / 小分子名称 / SMILES / RNA 亚型，
        # 就从这两个 Excel 里按原始 ID 补齐。
        "rna_info_path": r"data/raw/RNA.xlsx",
        "molecule_info_path": r"data/raw/Molecule.xlsx",

        # 五折 checkpoint 所在文件夹
        "weights_dir": r"checkpoints",

        # 输出目录
        "save_dir": "results",

        # 是否额外保存误差最小的 top 样本
        "save_top_n": 1328,
    }


# =========================
# 3. 元信息读取工具
# =========================
def to_plain_text(value):
    """
    把数据里的各种类型统一转成适合写入 CSV / Excel 的文本。
    兼容 str、数字、list、tuple、Tensor、bytes 等。
    """
    if value is None:
        return ""

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            value = value.item()
        else:
            value = value.tolist()

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    if isinstance(value, (list, tuple)):
        # RNA 序列如果是 ['A', 'U', 'G', ...]，直接拼起来
        if all(isinstance(x, str) and len(x) == 1 for x in value):
            return "".join(value)
        return "|".join([to_plain_text(x) for x in value])

    return str(value)


def _mapping_to_dict(obj):
    """尽量把 raw_info / pandas.Series / dict 转成普通 dict。"""
    if obj is None:
        return None

    if isinstance(obj, dict):
        return obj

    # pandas.Series / namedtuple 等通常有 to_dict
    if hasattr(obj, "to_dict"):
        try:
            d = obj.to_dict()
            if isinstance(d, dict):
                return d
        except Exception:
            pass

    return None


def _metadata_sources(item):
    """
    生成可检索的元信息来源。
    你的 .pt 样本里有 raw_info，因此这里不仅查 item 顶层，
    还会继续查 item['raw_info'] 里面的原始表字段。
    """
    sources = []

    item_dict = _mapping_to_dict(item)
    if item_dict is not None:
        sources.append(item_dict)

        for nested_key in ["raw_info", "raw", "meta", "metadata", "original_info", "source_info"]:
            nested = item_dict.get(nested_key, None)

            nested_dict = _mapping_to_dict(nested)
            if nested_dict is not None:
                sources.append(nested_dict)
                continue

            # 有些脚本会把 raw_info 保存成 list[dict] 或 tuple[dict]
            if isinstance(nested, (list, tuple)):
                for x in nested:
                    x_dict = _mapping_to_dict(x)
                    if x_dict is not None:
                        sources.append(x_dict)

    return sources


def get_first_existing(item, keys, default=""):
    """
    按优先级读取字段。
    同时兼容：
    1. 样本顶层字段，例如 r_id / m_id / rna_seq / frag_text；
    2. raw_info 内部字段，例如 RNA_ID / Molecule_ID / SMILES / RNA_name。
    """
    sources = _metadata_sources(item)

    for source in sources:
        # 先做精确匹配
        for key in keys:
            if key in source and source[key] is not None:
                value = to_plain_text(source[key]).strip()
                if value != "" and value.lower() != "nan":
                    return value

        # 再做大小写不敏感匹配
        lower_map = {str(k).lower(): k for k in source.keys()}
        for key in keys:
            real_key = lower_map.get(str(key).lower())
            if real_key is not None and source[real_key] is not None:
                value = to_plain_text(source[real_key]).strip()
                if value != "" and value.lower() != "nan":
                    return value

    return default


def safe_to_csv(df, path, encoding="utf-8-sig"):
    """
    保存 CSV。
    如果目标文件正被 Excel/WPS/Pandas 占用导致 PermissionError，
    自动换一个带时间戳的新文件名，避免程序中断。
    """
    try:
        df.to_csv(path, index=False, encoding=encoding)
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        new_path = f"{root}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        df.to_csv(new_path, index=False, encoding=encoding)
        print(f"\n⚠️ 原文件被占用，已自动改名保存：{new_path}")
        return new_path


def safe_to_excel(df, path):
    """保存 Excel；文件被占用时自动换名。"""
    try:
        df.to_excel(path, index=False)
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        new_path = f"{root}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        df.to_excel(new_path, index=False)
        print(f"\n⚠️ 原 Excel 文件被占用，已自动改名保存：{new_path}")
        return new_path


def normalize_id_for_match(value):
    """
    用于跨 .pt 和 Excel 匹配 ID。
    兼容 1、1.0、"1"、"1.0" 这类写法，避免因为 Excel 自动转数字导致匹配失败。
    """
    text = to_plain_text(value).strip()
    if text == "" or text.lower() == "nan":
        return ""

    # 去掉可能出现的引号和空格
    text = text.strip().strip("'").strip('"').strip()

    # Excel 里整数 ID 常被读成 1.0，这里统一还原成 1
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
    except Exception:
        pass

    return text


def is_blank_value(value):
    """判断输出表格中的某个单元格是否为空。"""
    text = to_plain_text(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def find_column_case_insensitive(df, candidates):
    """从 Excel 表中按候选列名寻找真实列名，先精确匹配，再忽略大小写匹配。"""
    columns = list(df.columns)

    for col in candidates:
        if col in columns:
            return col

    lower_map = {str(c).strip().lower(): c for c in columns}
    for col in candidates:
        real_col = lower_map.get(str(col).strip().lower())
        if real_col is not None:
            return real_col

    return None


def load_external_lookup(excel_path, id_candidates, field_candidates, table_name):
    """
    从 RNA.xlsx / Molecule.xlsx 读取 ID -> 元信息 的映射。

    field_candidates 形如：
    {
        "rna_name": ["RNA_name", "rna_name"],
        "rna_subtype": ["RNA_classes", "RNA_class"],
    }
    """
    if excel_path is None or str(excel_path).strip() == "":
        print(f"\n⚠️ 未设置 {table_name} 路径，跳过外部信息表补全。")
        return {}

    if not os.path.exists(excel_path):
        print(f"\n⚠️ 找不到 {table_name}: {excel_path}")
        print("将只使用 .pt 中已有的元信息。")
        return {}

    try:
        info_df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"\n⚠️ 读取 {table_name} 失败: {excel_path}")
        print(e)
        return {}

    print(f"\n✅ 已读取 {table_name}: {excel_path}")
    print(f"{table_name} 列名: {list(info_df.columns)}")

    id_col = find_column_case_insensitive(info_df, id_candidates)
    if id_col is None:
        print(f"⚠️ {table_name} 中没有找到 ID 列，候选列名为: {id_candidates}")
        return {}

    selected_cols = {}
    for out_col, candidates in field_candidates.items():
        real_col = find_column_case_insensitive(info_df, candidates)
        if real_col is not None:
            selected_cols[out_col] = real_col
        else:
            print(f"⚠️ {table_name} 中没有找到 {out_col} 对应列，候选列名为: {candidates}")

    lookup = {}
    for _, row in info_df.iterrows():
        key = normalize_id_for_match(row.get(id_col, ""))
        if key == "":
            continue

        lookup[key] = {
            out_col: to_plain_text(row.get(real_col, "")).strip()
            for out_col, real_col in selected_cols.items()
        }

    print(f"{table_name} 可用于匹配的 ID 数量: {len(lookup)}")
    return lookup


def fill_column_from_lookup(df, id_col, target_col, lookup, label):
    """只在 target_col 为空时，用 lookup 按 ID 补齐。"""
    if target_col not in df.columns or id_col not in df.columns or not lookup:
        return 0

    filled = 0
    new_values = []

    for _, row in df.iterrows():
        old_value = row[target_col]
        if not is_blank_value(old_value):
            new_values.append(old_value)
            continue

        key = normalize_id_for_match(row[id_col])
        value = lookup.get(key, {}).get(target_col, "")

        if not is_blank_value(value):
            new_values.append(value)
            filled += 1
        else:
            new_values.append(old_value)

    df[target_col] = new_values
    if filled > 0:
        print(f"{label}: 已补齐 {target_col} 共 {filled} 条。")
    return filled


def enrich_predictions_with_external_tables(df, args):
    """
    如果 .pt 中缺少名称 / SMILES / RNA 亚型，则从原始 RNA.xlsx 和 Molecule.xlsx 按 ID 补齐。

    当前你提供的两个表字段为：
    RNA.xlsx: RNA_ID, RNA_name, RNA_classes, RNA_Sequence
    Molecule.xlsx: Molecule_ID, Molecule_name, SMILES
    """
    rna_lookup = load_external_lookup(
        excel_path=args.get("rna_info_path", ""),
        id_candidates=[
            "RNA_ID", "rna_id", "R_ID", "r_id",
            "RNA_Metadata_ID", "RNA_metadata_id", "rna_metadata_id",
        ],
        field_candidates={
            "rna_name": ["RNA_name", "rna_name", "RNA_Name", "name"],
            "rna_subtype": [
                "RNA_classes", "RNA_class", "RNA_Class",
                "rna_classes", "rna_class",
                "RNA_subtype", "rna_subtype",
                "RNA_type", "rna_type",
            ],
            "rna_sequence": [
                "RNA_Sequence", "RNA_sequence", "rna_sequence",
                "RNA_seq", "rna_seq", "sequence", "seq",
            ],
        },
        table_name="RNA 信息表",
    )

    mol_lookup = load_external_lookup(
        excel_path=args.get("molecule_info_path", ""),
        id_candidates=[
            "Molecule_ID", "molecule_id", "M_ID", "m_id",
            "Molecule_Metadata_ID", "Molecule_metadata_id", "molecule_metadata_id",
            "Mol_ID", "mol_id", "Compound_ID", "compound_id",
        ],
        field_candidates={
            "mol_name": [
                "Molecule_name", "molecule_name", "Molecule_Name",
                "mol_name", "Mol_name", "compound_name", "Compound_name", "name",
            ],
            "smiles": [
                "SMILES", "smiles", "Smiles",
                "molecule_smiles", "Molecule_SMILES",
                "canonical_smiles", "Canonical_SMILES",
            ],
        },
        table_name="Molecule 信息表",
    )

    print("\n开始用原始 RNA.xlsx / Molecule.xlsx 补齐输出表格元信息...")
    total_filled = 0
    total_filled += fill_column_from_lookup(df, "rna_id", "rna_name", rna_lookup, "RNA 信息表")
    total_filled += fill_column_from_lookup(df, "rna_id", "rna_subtype", rna_lookup, "RNA 信息表")
    total_filled += fill_column_from_lookup(df, "rna_id", "rna_sequence", rna_lookup, "RNA 信息表")
    total_filled += fill_column_from_lookup(df, "mol_id", "mol_name", mol_lookup, "Molecule 信息表")
    total_filled += fill_column_from_lookup(df, "mol_id", "smiles", mol_lookup, "Molecule 信息表")

    if total_filled == 0:
        print("⚠️ 没有补齐任何字段。请检查 .pt 中的 r_id / m_id 是否与两个 Excel 的 ID 一致。")

    return df


# 原始 RNA ID：优先使用原始表中的 RNA_ID / RNA_Metadata_ID 等，不再强行改成 RNA_0 这种编号
RNA_ID_KEYS = [
    # 你当前 .pt 顶层实际字段
    "r_id", "R_ID", "R_id",

    # 常见原始表字段
    "rna_original_id", "orig_rna_id", "original_rna_id",
    "RNA_ID", "rna_ID", "rna_id",
    "RNA_Metadata_ID", "RNA_metadata_id", "rna_metadata_id",
    "RNAId", "RNA_Id",
]

# 原始小分子 ID：优先使用原始表中的 Molecule_ID / Molecule_Metadata_ID 等
MOL_ID_KEYS = [
    # 你当前 .pt 顶层实际字段
    "m_id", "M_ID", "M_id",

    # 常见原始表字段
    "mol_original_id", "orig_mol_id", "original_mol_id",
    "Molecule_ID", "molecule_ID", "molecule_id",
    "Mol_ID", "mol_ID", "mol_id",
    "Molecule_Metadata_ID", "Molecule_metadata_id", "molecule_metadata_id", "mol_metadata_id",
    "MoleculeId", "Molecule_Id", "Compound_ID", "compound_id",
]

RNA_SEQ_KEYS = [
    "rna_seq", "RNA_seq",
    "rna_sequence", "RNA_sequence", "RNA_Sequence", "sequence", "seq",
]

SMILES_KEYS = [
    "smiles", "SMILES", "Smiles",
    "mol_smiles", "Mol_SMILES",
    "molecule_smiles", "Molecule_SMILES",
    "compound_smiles", "Compound_SMILES",
    "canonical_smiles", "Canonical_SMILES",
]

RNA_SUBTYPE_KEYS = [
    "rna_subtype", "RNA_subtype", "RNA_Subtype",
    "rna_type", "RNA_type", "RNA_Type",
    "subtype", "Subtype", "type",
    "RNA_classes", "rna_classes",
    "RNA_class", "rna_class", "RNA_category", "rna_category",
]

RNA_NAME_KEYS = [
    "rna_name", "RNA_name", "RNA_Name",
    "rna_names", "RNA_names",
    "rna_title", "RNA_title",
    "RNA", "rna", "name",
]

MOL_NAME_KEYS = [
    "mol_name", "Mol_name", "Mol_Name",
    "molecule_name", "Molecule_name", "Molecule_Name",
    "compound_name", "Compound_name", "Compound_Name",
    "ligand_name", "Ligand_name", "Ligand_Name",
    "Molecule", "molecule", "Compound", "compound",
]


# =========================
# 4. 数据集
# =========================
class TripleTowerDataset(Dataset):
    def __init__(self, data_list):
        self.raw_data = data_list

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]

        return {
            "rna_orig": item["rna_orig"].half(),
            "rna_window": item["rna_window"].half(),
            "mol_global": item["mol_global"].half(),
            "mol_frags": item["mol_frags"].half(),
            "label": item["label"].to(torch.float32),

            # ===== 原始信息，用于保存预测结果 =====
            # RNA / 小分子使用原始 ID，而不是重新生成 RNA_0、Mol_0
            "rna_id": get_first_existing(item, RNA_ID_KEYS, default=""),
            "mol_id": get_first_existing(item, MOL_ID_KEYS, default=""),

            # 表格中额外保存的信息
            "rna_seq": get_first_existing(item, RNA_SEQ_KEYS, default=""),
            "smiles": get_first_existing(item, SMILES_KEYS, default=""),
            "rna_subtype": get_first_existing(item, RNA_SUBTYPE_KEYS, default=""),
            "rna_name": get_first_existing(item, RNA_NAME_KEYS, default=""),
            "mol_name": get_first_existing(item, MOL_NAME_KEYS, default=""),

            # 如果你的数据里有 fragment 文本，也一起保留，方便检查
            "frag_text": get_first_existing(item, ["frag_text", "frags_text", "mol_frag_text"], default=""),
        }


# =========================
# 5. collate_fn：处理变长 RNA / fragment
# =========================
def collate_fn(batch):
    rna_o = pad_sequence(
        [item["rna_orig"] for item in batch],
        batch_first=True
    )

    rna_w = pad_sequence(
        [item["rna_window"] for item in batch],
        batch_first=True
    )

    frags = pad_sequence(
        [item["mol_frags"] for item in batch],
        batch_first=True
    )

    return {
        "rna_o": rna_o,
        "rna_o_mask": (rna_o.sum(dim=-1) == 0),

        "rna_w": rna_w,
        "rna_w_mask": (rna_w.sum(dim=-1) == 0),

        "frag_embeds": frags,
        "frag_mask": (frags.sum(dim=-1) == 0),

        "global_mol_embed": torch.stack(
            [item["mol_global"] for item in batch]
        ),

        "labels": torch.stack(
            [item["label"] for item in batch]
        ),

        # 元信息
        "rna_ids": [item["rna_id"] for item in batch],
        "mol_ids": [item["mol_id"] for item in batch],
        "rna_seqs": [item["rna_seq"] for item in batch],
        "smiles": [item["smiles"] for item in batch],
        "rna_subtypes": [item["rna_subtype"] for item in batch],
        "rna_names": [item["rna_name"] for item in batch],
        "mol_names": [item["mol_name"] for item in batch],
        "frag_texts": [item["frag_text"] for item in batch],
    }


# =========================
# 6. 指标计算
# =========================
def calculate_metrics(labels, preds):
    labels = np.asarray(labels).reshape(-1)
    preds = np.asarray(preds).reshape(-1)

    mae = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    r2 = r2_score(labels, preds)

    if len(np.unique(labels)) > 1:
        pcc, _ = pearsonr(labels, preds)
        scc, _ = spearmanr(labels, preds)
    else:
        pcc, scc = 0.0, 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "PCC": pcc,
        "SCC": scc,
    }


# =========================
# 7. 自然排序 checkpoint
# =========================
def natural_sort_key(path):
    """
    让 fold1, fold2, fold10 按数字顺序排列。
    虽然平均值不受顺序影响，但打印和检查更清楚。
    """
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


# =========================
# 8. 加载 checkpoint
# =========================
def load_model_weights(model, ckpt_path, device):
    """
    兼容两种 checkpoint 格式：
    1. 直接保存的 state_dict
    2. 字典格式，里面包含 model_state_dict 或 state_dict
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # 兼容 DataParallel 保存时出现的 module. 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k.replace("module.", "", 1)] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


# =========================
# 9. 单个模型预测全量数据
# =========================
def predict_with_one_checkpoint(model, loader, device):
    model.eval()

    preds = []
    labels = []

    # 保存元信息
    rna_ids = []
    mol_ids = []
    rna_seqs = []
    smiles_list = []
    rna_subtypes = []
    rna_names = []
    mol_names = []
    frag_texts = []

    use_cuda_amp = device.type == "cuda"
    autocast_ctx = torch.amp.autocast("cuda") if use_cuda_amp else nullcontext()

    with torch.no_grad():
        for b in tqdm(loader, desc="Inference", leave=False):
            with autocast_ctx:
                res = model(
                    rna_orig=b["rna_o"].to(device, dtype=torch.float32),
                    rna_window=b["rna_w"].to(device, dtype=torch.float32),
                    mol_global=b["global_mol_embed"].to(device, dtype=torch.float32),
                    mol_frags=b["frag_embeds"].to(device, dtype=torch.float32),
                    rna_orig_mask=b["rna_o_mask"].to(device),
                    rna_window_mask=b["rna_w_mask"].to(device),
                    frag_mask=b["frag_mask"].to(device),
                )

            batch_preds = res["out"].detach().cpu().numpy().reshape(-1)
            batch_labels = b["labels"].detach().cpu().numpy().reshape(-1)

            preds.extend(batch_preds)
            labels.extend(batch_labels)

            rna_ids.extend(b["rna_ids"])
            mol_ids.extend(b["mol_ids"])
            rna_seqs.extend(b["rna_seqs"])
            smiles_list.extend(b["smiles"])
            rna_subtypes.extend(b["rna_subtypes"])
            rna_names.extend(b["rna_names"])
            mol_names.extend(b["mol_names"])
            frag_texts.extend(b["frag_texts"])

    return {
        "preds": np.asarray(preds),
        "labels": np.asarray(labels),

        "rna_ids": rna_ids,
        "mol_ids": mol_ids,
        "rna_seqs": rna_seqs,
        "smiles": smiles_list,
        "rna_subtypes": rna_subtypes,
        "rna_names": rna_names,
        "mol_names": mol_names,
        "frag_texts": frag_texts,
    }


# =========================
# 10. 五折 checkpoint 平均预测所有数据
# =========================
def run_ensemble_inference(args):
    seed_everything(args.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")

    # ---------- 加载数据 ----------
    data_pack = torch.load(args["data_path"], weights_only=True)

    if isinstance(data_pack, dict) and "dataset" in data_pack:
        raw_samples = data_pack["dataset"]
    else:
        raw_samples = data_pack

    print(f"数据集样本数: {len(raw_samples)}")

    # ---------- 自动读取输入维度 ----------
    sample = raw_samples[0]
    args["rinalmo_dim"] = sample["rna_orig"].shape[-1]
    args["chemberta_dim"] = sample["mol_global"].shape[-1]
    args["melm_dim"] = sample["mol_frags"].shape[-1]

    print(f"rinalmo_dim: {args['rinalmo_dim']}")
    print(f"chemberta_dim: {args['chemberta_dim']}")
    print(f"melm_dim: {args['melm_dim']}")

    # 打印样本中已有字段，方便确认原始 ID / 名称 / SMILES 是否真的已经写进 pt
    print("\n当前 .pt 单个样本包含的字段：")
    print(list(sample.keys()))

    if isinstance(sample, dict) and "raw_info" in sample:
        raw_info_dict = _mapping_to_dict(sample["raw_info"])
        if raw_info_dict is not None:
            print("\n当前 raw_info 里面包含的字段：")
            print(list(raw_info_dict.keys()))

    # ---------- DataLoader ----------
    full_dataset = TripleTowerDataset(raw_samples)

    full_loader = DataLoader(
        full_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ---------- 查找 checkpoint ----------
    weight_files = sorted(
        glob.glob(os.path.join(args["weights_dir"], "*.pt")),
        key=natural_sort_key,
    )

    if len(weight_files) == 0:
        print(f"❌ 没有在该文件夹找到 .pt checkpoint: {args['weights_dir']}")
        return None

    print(f"\n检测到 {len(weight_files)} 个 checkpoint：")
    for p in weight_files:
        print(" -", os.path.basename(p))

    if len(weight_files) != 5:
        print(f"\n⚠️ 注意：当前检测到 {len(weight_files)} 个 checkpoint，不是 5 个。")
        print("如果你确认这些都是需要平均的 checkpoint，可以继续使用。")

    # ---------- 循环五折模型，预测全量数据 ----------
    all_fold_preds = []
    final_labels = None

    final_rna_ids = None
    final_mol_ids = None
    final_rna_seqs = None
    final_smiles = None
    final_rna_subtypes = None
    final_rna_names = None
    final_mol_names = None
    final_frag_texts = None

    model = MSGA_RNA(args).to(device)

    for fold_i, ckpt_path in enumerate(weight_files, start=1):
        print(f"\n===== 使用 checkpoint {fold_i}/{len(weight_files)} 预测 =====")
        print(f"读取权重: {os.path.basename(ckpt_path)}")

        load_model_weights(model, ckpt_path, device)

        one_result = predict_with_one_checkpoint(
            model=model,
            loader=full_loader,
            device=device,
        )

        all_fold_preds.append(one_result["preds"])

        if final_labels is None:
            final_labels = one_result["labels"]

            final_rna_ids = one_result["rna_ids"]
            final_mol_ids = one_result["mol_ids"]
            final_rna_seqs = one_result["rna_seqs"]
            final_smiles = one_result["smiles"]
            final_rna_subtypes = one_result["rna_subtypes"]
            final_rna_names = one_result["rna_names"]
            final_mol_names = one_result["mol_names"]
            final_frag_texts = one_result["frag_texts"]

    # ---------- 五折平均 ----------
    all_fold_preds = np.stack(all_fold_preds, axis=0)

    avg_preds = np.mean(all_fold_preds, axis=0)
    std_preds = np.std(all_fold_preds, axis=0)
    abs_errors = np.abs(avg_preds - final_labels)

    # ---------- 计算整体指标 ----------
    metrics = calculate_metrics(final_labels, avg_preds)

    print("\n" + "=" * 40)
    print(" 五折 checkpoint 平均预测全量数据结果 ")
    print("=" * 40)
    print(f"MAE :  {metrics['MAE']:.4f}")
    print(f"RMSE:  {metrics['RMSE']:.4f}")
    print(f"R2  :  {metrics['R2']:.4f}")
    print(f"PCC :  {metrics['PCC']:.4f}")
    print(f"SCC :  {metrics['SCC']:.4f}")
    print("=" * 40)

    # ---------- 整理所有预测结果 ----------
    result_rows = []

    for i in range(len(final_labels)):
        row = {
            "sample_index": i,

            # 原始 ID
            "rna_id": final_rna_ids[i],
            "mol_id": final_mol_ids[i],

            # 要展示在表格中的原始信息
            "rna_sequence": final_rna_seqs[i],
            "smiles": final_smiles[i],
            "rna_subtype": final_rna_subtypes[i],
            "rna_name": final_rna_names[i],
            "mol_name": final_mol_names[i],
            "frag_text": final_frag_texts[i],

            # 真实值与预测值
            "true_affinity": float(final_labels[i]),
            "pred_affinity_avg": float(avg_preds[i]),
            "pred_affinity_std": float(std_preds[i]),
            "abs_error": float(abs_errors[i]),
        }

        # 保存每个 fold 的单独预测值
        for fold_i in range(len(weight_files)):
            row[f"pred_fold_{fold_i + 1}"] = float(all_fold_preds[fold_i, i])

        result_rows.append(row)

    df_all = pd.DataFrame(result_rows)

    # 表格列顺序：先展示用户关心的信息，再展示预测结果
    fixed_cols = [
        "sample_index",
        "rna_id",
        "mol_id",
        "rna_name",
        "mol_name",
        "rna_subtype",
        "rna_sequence",
        "smiles",
        "frag_text",
        "true_affinity",
        "pred_affinity_avg",
        "pred_affinity_std",
        "abs_error",
    ]
    fold_cols = [f"pred_fold_{i + 1}" for i in range(len(weight_files))]
    df_all = df_all[fixed_cols + fold_cols]

    # 如果 .pt 中缺少名称 / SMILES / RNA 亚型 / RNA 序列，
    # 就从 data/raw/RNA.xlsx 和 Molecule.xlsx 按原始 ID 补齐。
    df_all = enrich_predictions_with_external_tables(df_all, args)

    # 如果补齐后仍然没有某些字段，这里提醒一下
    required_show_cols = ["rna_id", "mol_id", "rna_sequence", "smiles", "rna_subtype", "rna_name", "mol_name"]
    empty_cols = [c for c in required_show_cols if df_all[c].astype(str).str.len().sum() == 0]
    if empty_cols:
        print("\n⚠️ 以下列在当前 .pt 数据中没有读到内容，因此输出表格中会为空：")
        print(empty_cols)
        print("请检查数据处理脚本是否已经把这些字段写入 final_data_random.pt。")

    # ---------- 保存结果 ----------
    dataset_tag = os.path.basename(args["data_path"]).replace(".pt", "")
    save_dir = os.path.join(args["save_dir"], dataset_tag, "ensemble_all_data")
    os.makedirs(save_dir, exist_ok=True)

    all_save_path = os.path.join(save_dir, "ensemble_all_predictions.csv")
    all_save_path = safe_to_csv(df_all, all_save_path)

    # 同时保存 Excel，方便直接查看表格
    all_excel_path = os.path.join(save_dir, "ensemble_all_predictions.xlsx")
    try:
        all_excel_path = safe_to_excel(df_all, all_excel_path)
        print(f"\n✅ 全量数据五折平均预测结果 Excel 已保存至:")
        print(all_excel_path)
    except Exception as e:
        print("\n⚠️ Excel 保存失败，只保存 CSV。原因：")
        print(e)

    print(f"\n✅ 全量数据五折平均预测结果 CSV 已保存至:")
    print(all_save_path)

    # ---------- 保存 top_n 最准样本 ----------
    top_n = args.get("save_top_n", None)

    if top_n is not None and top_n > 0:
        df_top = df_all.sort_values("abs_error", ascending=True).head(top_n)

        top_save_path = os.path.join(save_dir, f"ensemble_top_{top_n}.csv")
        top_save_path = safe_to_csv(df_top, top_save_path)

        top_excel_path = os.path.join(save_dir, f"ensemble_top_{top_n}.xlsx")
        try:
            top_excel_path = safe_to_excel(df_top, top_excel_path)
            print(f"\n✅ 预测误差最小的 Top {top_n} 样本 Excel 已保存至:")
            print(top_excel_path)
        except Exception as e:
            print("\n⚠️ Top Excel 保存失败，只保存 CSV。原因：")
            print(e)

        print(f"\n✅ 预测误差最小的 Top {top_n} 样本 CSV 已保存至:")
        print(top_save_path)

        print("\nTop 10 预览：")
        preview_cols = [
            "rna_id",
            "mol_id",
            "rna_name",
            "mol_name",
            "rna_subtype",
            "true_affinity",
            "pred_affinity_avg",
            "pred_affinity_std",
            "abs_error",
        ]
        print(df_top[preview_cols].head(10))

    # ---------- 保存整体指标 ----------
    metrics_save_path = os.path.join(save_dir, "ensemble_metrics.csv")
    metrics_save_path = safe_to_csv(pd.DataFrame([metrics]), metrics_save_path)

    metrics_excel_path = os.path.join(save_dir, "ensemble_metrics.xlsx")
    try:
        metrics_excel_path = safe_to_excel(pd.DataFrame([metrics]), metrics_excel_path)
    except Exception:
        pass

    print(f"\n✅ 整体指标已保存至:")
    print(metrics_save_path)

    return df_all, metrics


# =========================
# 11. 主程序
# =========================
if __name__ == "__main__":
    args = get_args()
    run_ensemble_inference(args)
