"""Pack the released R-SIM feature tensors in the original interaction order."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch


def main():
    data = Path("data")
    labels = data / "raw/RNA-Molecule.xlsx"
    # The saved fold indices refer to rows of this exact interaction table.
    expected = "e0eb1b746518ea3121285a6d0bba2378cba297a898e355430e03226d241ad4ab"
    if hashlib.sha256(labels.read_bytes()).hexdigest() != expected:
        raise ValueError("The supplied folds require the original RNA-Molecule.xlsx.")
    rows = pd.read_excel(labels)
    cache = data / "features"
    features = {
        name: torch.load(cache / f"{name}.pt", map_location="cpu", weights_only=True)
        for name in ("rna_orig", "rna_window", "mol_global", "mol_frags")
    }
    fragments = json.loads((cache / "fragment_smiles.json").read_text())
    samples = []
    for row in rows.itertuples(index=False):
        rna_id, mol_id = str(int(row.RNA_ID)), str(int(row.Molecule_ID))
        samples.append({
            "r_id": rna_id,
            "m_id": mol_id,
            "rna_orig": features["rna_orig"][rna_id],
            "rna_window": features["rna_window"][rna_id],
            "mol_global": features["mol_global"][mol_id],
            "mol_frags": features["mol_frags"][mol_id],
            "frag_text": fragments[mol_id],
            "label": torch.tensor(row.pKd, dtype=torch.float32),
        })

    folds = json.loads((data / "folds.json").read_text())
    output = data / "processed/final_data_random.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dataset": samples, "folds": folds}, output)
    print(f"Packed {len(samples)} interactions and {len(folds)} folds in {output}")


if __name__ == "__main__":
    main()
