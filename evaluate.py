"""Evaluate each supplied checkpoint on its original held-out test partition."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from training import TripleTowerDataset, collate_fn, evaluate_model, get_args, seed_everything
from models.model import MSGA_RNA


def main():
    args = get_args()
    seed_everything(args["seed"])
    pack = torch.load(args["data_path"], map_location="cpu", weights_only=True)
    dataset = TripleTowerDataset(pack["dataset"])
    example = dataset[0]
    args.update(rinalmo_dim=example["rna_orig"].shape[-1],
                chemberta_dim=example["mol_global"].shape[-1],
                melm_dim=example["mol_frags"].shape[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSGA_RNA(args).to(device)
    rows = []
    for split in pack["folds"]:
        fold = split["fold"]
        path, = Path("checkpoints").glob(f"*_fold{fold}.pt")
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        indices = split["test_idx"]
        loader = DataLoader(Subset(dataset, indices), batch_size=args["batch_size"], collate_fn=collate_fn)
        scores = evaluate_model(model, loader, device, split_name="test")
        rows.append({"fold": fold, "pairs": len(indices), **scores})

    table = pd.DataFrame(rows)
    output = Path("results/held_out_folds.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print("Mean fold RMSE", np.mean(table["RMSE"]))


if __name__ == "__main__":
    main()
