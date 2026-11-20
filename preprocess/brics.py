import pandas as pd
import os
import torch
from rdkit import Chem
from rdkit.Chem import BRICS
from tqdm import tqdm


def process_data(data_dir, output_path):
    # 文件路径 - 现在主要使用 Molecule.xlsx
    mol_file = os.path.join(data_dir, 'Molecule.xlsx')

    print("📖 正在读取 Molecule Excel 数据...")
    mol_df = pd.read_excel(mol_file)

    # 1. 拆分分子碎片并构建数据集
    dataset = []
    print("✂️ 开始 BRICS 拆分并构建数据...")
    for _, row in tqdm(mol_df.iterrows(), total=len(mol_df), desc="Processing Molecules"):
        m_id = int(row['Molecule_ID'])
        smiles = str(row['SMILES'])

        # RDKit 处理与碎片拆分
        mol = Chem.MolFromSmiles(smiles)
        frags = list(BRICS.BRICSDecompose(mol)) if mol else [smiles]

        # 按照要求的顺序构建字典（对应列顺序）
        dataset.append({
            'molecule_id': m_id,  # 第一列
            'mol_frags': frags,  # 第二列
            'full_smiles': smiles  # 第三列
        })

    # 2. 保存为 .pt 文件
    torch.save(dataset, output_path)

    # 3. 保存为 .csv 文件 (仅包含这三列)
    csv_output_path = output_path.replace('.pt', '.csv')
    df_result = pd.DataFrame(dataset)

    # 将 list 类型的碎片转换为字符串，用 "|" 分隔，方便查看
    df_result['mol_frags'] = df_result['mol_frags'].apply(lambda x: "|".join(x))

    # 按照要求的列顺序重新排列并保存
    df_result = df_result[['molecule_id', 'mol_frags', 'full_smiles']]
    df_result.to_csv(csv_output_path, index=False, encoding='utf-8-sig')

    print(f"🎉 处理完成！共 {len(dataset)} 个分子")
    print(f"💾 PT 文件已保存 (含 ID/碎片/整体): {output_path}")
    print(f"📊 CSV 文件已保存 (三列顺序): {csv_output_path}")


if __name__ == "__main__":
    DATA_DIR = r'data/raw'
    out_dir = 'data/processed/extracted'
    os.makedirs(out_dir, exist_ok=True)
    process_data(DATA_DIR, os.path.join(out_dir, 'intermediate_data.pt'))