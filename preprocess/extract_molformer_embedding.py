import torch
import pandas as pd
import os
import gc
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "data/processed/extracted"

DATA_DIR = r'data/raw'
MOL_EXCEL = os.path.join(DATA_DIR, 'Molecule.xlsx')
FRAG_DATA_PATH = os.path.join(OUTPUT_DIR, 'intermediate_data.pt')

# ✅ 只跑 MoLFormer
MODEL_DIRS = [
    "MoLFormer-c3-1.1B"
]


class MolFeatureExtractor:
    def __init__(self, model_path):
        print(f"\n🚀 正在加载模型 {model_path}...")

        # tokenizer（必须 trust_remote_code）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        print("💡 使用 MoLFormer 稳定加载模式...")

        # 模型加载（不使用 half）
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True
        ).to(DEVICE)

        self.model.eval()
        self.hidden_size = self.model.config.hidden_size
        print(f"📏 模型维度: {self.hidden_size}")

    @torch.no_grad()
    def get_embedding(self, smiles):
        """提取特征向量"""
        if not smiles or pd.isna(smiles) or not isinstance(smiles, str):
            return torch.zeros(self.hidden_size)

        inputs = self.tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        ).to(DEVICE)

        outputs = self.model(**inputs)

        # ✅ mean pooling（比 CLS 稳定）
        embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding.squeeze().cpu()


def main():
    print("📖 读取数据中...")

    if not os.path.exists(MOL_EXCEL):
        raise FileNotFoundError(MOL_EXCEL)

    df_mol = pd.read_excel(MOL_EXCEL)

    id_col = [c for c in df_mol.columns if 'ID' in str(c).upper()][0]
    smi_col = [c for c in df_mol.columns if 'SMILES' in str(c).upper()][0]

    mol_info = {
        str(row[id_col]): row[smi_col]
        for _, row in df_mol.iterrows()
    }

    print(f"📂 加载碎片文件: {FRAG_DATA_PATH}")
    frag_raw = torch.load(FRAG_DATA_PATH)

    mol_to_frags = {}
    for item in frag_raw:
        m_id = str(item['molecule_id'])
        mol_to_frags[m_id] = item['mol_frags']

    for model_folder in MODEL_DIRS:

        # ✅ 直接用 HF 模型（关键）
        if "MoLFormer" in model_folder:
            model_path = "ibm/MoLFormer-XL-both-10pct"
        else:
            continue

        save_path_global = os.path.join(OUTPUT_DIR, "mol_global.pt")
        save_path_frags = os.path.join(OUTPUT_DIR, "mol_frags.pt")

        extractor = MolFeatureExtractor(model_path)

        results_global = {}
        results_frags = {}

        print(f"🧬 正在提取 [{model_folder}] 的特征...")

        for m_id, full_smi in tqdm(mol_info.items(), desc=f"Processing {model_folder}"):

            # --- 1. Global embedding ---
            v_global = extractor.get_embedding(full_smi)
            results_global[m_id] = v_global

            # --- 2. Fragment embeddings ---
            frags = mol_to_frags.get(m_id, [])

            if not frags:
                v_frags = torch.zeros((1, extractor.hidden_size))
            else:
                v_frags = torch.stack([
                    extractor.get_embedding(f) for f in frags
                ])

            results_frags[m_id] = v_frags

        # --- 保存 ---
        torch.save(results_global, save_path_global)
        torch.save(results_frags, save_path_frags)

        print(f"✅ [{model_folder}] 完成！")

        # --- 清理显存 ---
        del extractor
        gc.collect()
        torch.cuda.empty_cache()

    print("\n🎉 特征提取全部跑通！")


if __name__ == "__main__":
    main()